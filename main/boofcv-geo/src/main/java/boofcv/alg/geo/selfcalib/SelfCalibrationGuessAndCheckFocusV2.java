/*
 * Copyright (c) 2011-2020, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://boofcv.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boofcv.alg.geo.selfcalib;

import boofcv.abst.geo.Triangulate2ViewsMetric;
import boofcv.alg.geo.MultiViewOps;
import boofcv.alg.geo.PerspectiveOps;
import boofcv.factory.geo.FactoryMultiView;
import boofcv.struct.calib.CameraPinhole;
import boofcv.struct.geo.AssociatedTriple;
import georegression.struct.point.Point2D_F64;
import georegression.struct.point.Point3D_F64;
import georegression.struct.point.Vector3D_F64;
import georegression.struct.se.Se3_F64;
import georegression.transform.se.SePointOps_F64;
import lombok.Getter;
import lombok.Setter;
import org.ddogleg.struct.FastQueue;
import org.ddogleg.struct.VerbosePrint;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * <p>
 *     Computes the best projective to metric 4x4 rectifying homography matrix by guessing different values
 *     for focal lengths of the first two views. Focal lengths are guessed using a log scale. Skew and image center
 *     are both assumed to be known and have to be specified by the user. This strategy shows better convergence
 *     than methods which attempt to guess the focal length using linear or gradient descent approaches due
 *     to the vast number of local minima in the search space. Non-linear refinement is highly recommended after
 *     using this algorithm due to its approximate nature.
 * </p>
 * <p>
 *     NOTE: Performance on noise free synthetic data replicates paper claims. Have not been able to replicate
 *     performance on real data. Authors were contacted for a reference implementation and was told source code
 *     is not publicly available.
 * </p>
 *
 * <ul>
 *     <li>if sameFocus is set to true then the first two views are assumed to have approximately the same focal length</li>
 *     <li>Internally, the plane at infinity is computed using the known intrinsic parameters.</li>
 *     <li>Rectifying homography is computed using known K in first view and plane at infinity. lambda of 1 is assumed</li>
 * </ul>
 *
 * Changes from paper:
 * <ol>
 *     <li>Extracting K using absolute quadratic instead of rectifying homography</li>
 * </ol>
 *
 * @see EstimatePlaneAtInfinityGivenK
 *
 * <p>
 * <li>Gherardi, Riccardo, and Andrea Fusiello. "Practical autocalibration."
 * European Conference on Computer Vision. Springer, Berlin, Heidelberg, 2010.</li>
 * </p>
 *
 * @author Peter Abeles
 */
public class SelfCalibrationGuessAndCheckFocusV2 implements VerbosePrint {

	// used to estimate the plane at infinity
	EstimatePlaneAtInfinityGivenK estimatePlaneInf = new EstimatePlaneAtInfinityGivenK();
	Vector3D_F64 planeInf = new Vector3D_F64();

	// if true the first two cameras are assumed to have the same or approximately the same focus length
	private @Getter @Setter boolean sameFocus = true;

	// intrinsic camera calibration matrix for view 1
	DMatrixRMaj K1 = new DMatrixRMaj(3,3);

	// Work space for view 1 projective matrix
	DMatrixRMaj P1 = new DMatrixRMaj(3,4);
	DMatrixRMaj P2 = new DMatrixRMaj(3,4);
	DMatrixRMaj P3 = new DMatrixRMaj(3,4);

	CameraPinhole intrinsic1 = new CameraPinhole();
	CameraPinhole intrinsic2 = new CameraPinhole();
	CameraPinhole intrinsic3 = new CameraPinhole();

	@Getter double bestFocus1, bestFocus2;

	Se3_F64 view_1_to_1 = new Se3_F64();
	Se3_F64 view_1_to_2 = new Se3_F64();
	Se3_F64 view_1_to_3 = new Se3_F64();

	DMatrixRMaj K = new DMatrixRMaj(3,3);

	// projective to metric homography
	DMatrixRMaj H = new DMatrixRMaj(4,4);
	DMatrixRMaj bestH = new DMatrixRMaj(4,4);

	// Absolute dual quadratic
//	DMatrixRMaj Q = new DMatrixRMaj(4,4);

	// Defines which focus lengths are sampled based on a log scale
	// Note that image has been normalized and 1.0 = focal length of image diagonal
	double sampleMin=0.3,sampleMax=3;
	int numSamples=50;
	double[] errors = new double[numSamples];
	double[] errors1 = new double[numSamples];
	double[] errors2 = new double[numSamples];

	DMatrixRMaj tmp = new DMatrixRMaj(3,3);

	// Is the best score at a local minimum? If not that means it probably diverged
	boolean localMinimum;

	int foundInvalid = 0;

	// if not null debug info is printed
	PrintStream verbose;

	// reference to input parameter. Observed pixel associations
	List<AssociatedTriple> matchesPixels;
	FastQueue<AssociatedTriple> matchesNorm = new FastQueue<>(AssociatedTriple::new);

	/**
	 * Specifies how focal lengths are sampled on a log scale. Remember 1.0 = nominal length
	 *
	 * @param min min value. 0.3 is default
	 * @param max max value. 3.0 is default
	 * @param total Number of sample points. 50 is default
	 */
	public void setSampling( double min , double max , int total ) {
		this.sampleMin = min;
		this.sampleMax = max;
		this.numSamples = total;
		this.errors = new double[numSamples];
	}

	public boolean process(DMatrixRMaj camera1, DMatrixRMaj camera2, DMatrixRMaj camera3, List<AssociatedTriple> matches ) {
		this.P1.set(camera1);
		this.P2.set(camera2);
		this.P3.set(camera3);

		this.matchesPixels = matches;

		// Force the first camera to be identity
		MultiViewOps.projectiveToIdentityH(P1,H);
		CommonOps_DDRM.mult(P1,H,tmp); P1.set(tmp);
		CommonOps_DDRM.mult(P2,H,tmp); P2.set(tmp);
		CommonOps_DDRM.mult(P3,H,tmp); P3.set(tmp);

//		P1.print();
//		P2.print();

		// Find the best combinations of focal lengths
		double bestScore;
		if( sameFocus ) {
			bestScore = findBestFocusOne(P2);
		} else {
			bestScore = findBestFocusTwo(P2);
		}

		bestH.print();

		// if it's not at a local minimum it almost certainly failed
		return bestScore != Double.MAX_VALUE && localMinimum;
	}

	private double findBestFocusOne(DMatrixRMaj P2) {
		localMinimum = false;

		// coeffients for linear to log scale
		double b = Math.log(sampleMax/sampleMin)/(numSamples-1);
		double bestError = Double.MAX_VALUE;
		int bestIndex = -1;

		for (int i = 0; i < numSamples; i++) {
			double f = sampleMin*Math.exp(b*i)*800;

			if( !computeRectifyH(f,f,P2,H)) {
				errors[i] = Double.NaN;
				continue;
			}
//			MultiViewOps.rectifyHToAbsoluteQuadratic(H,Q);

			double error = computeError();
			errors[i] = error;

			if( error < bestError ) {
				bestError = error;
				bestH.set(H);
				bestIndex = i;
				bestFocus1 = f;
				bestFocus2 = f;
			}

			if( verbose != null ) {
				verbose.printf("[%3d] f=%5.2f score=%f invalid=%d\n",i,f,error,foundInvalid);
			}
		}

		if (bestIndex > 0 && bestIndex < numSamples - 1) {
			localMinimum = bestError < errors[bestIndex - 1] && bestError < errors[bestIndex + 1];
		}

		return bestError;
	}

	private double findBestFocusTwo(DMatrixRMaj P2) {
		localMinimum = false;

		// coefficients for linear to log scale
		double b = Math.log(sampleMax/sampleMin)/(numSamples-1);
		double bestError = Double.MAX_VALUE;
		int bestInvalid = Integer.MAX_VALUE;

		Arrays.fill(errors1,0);
		Arrays.fill(errors2,0);

		for (int idx1 = 0; idx1 < numSamples; idx1++) {
			double f1 =sampleMin*Math.exp(b*idx1)*800;
			if( idx1 == 20 )
				f1 = 600;

			boolean minimumChanged = false;
			int bestIndex = -1;

			for (int idx2 = 0; idx2 < numSamples; idx2++) {
				double f2 =sampleMin*Math.exp(b*idx2)*800;
				if( idx2 == 20 )
					f2 = 600;

				if( !computeRectifyH(f1,f2,P2,H)) {
					errors1[idx1] = Double.NaN;
					errors2[idx2] = Double.NaN;
					continue;
				}
//				MultiViewOps.rectifyHToAbsoluteQuadratic(H,Q);

				double error = computeError();
				errors1[idx1] += error;
				errors2[idx2] += error;

				boolean better = false;
				if( foundInvalid < bestInvalid ) {
					better = true;
				} else if( foundInvalid == bestInvalid && error < bestError) {
					better = true;
				}

				if( better ) {
					minimumChanged = true;
					bestIndex = idx2;
					bestError = error;
					bestInvalid = foundInvalid;
					bestH.set(H);
					bestFocus1 = f1;
					bestFocus2 = f2;
				}

				if( verbose != null ) {
					verbose.printf("[%3d,%3d] f1=%5.2f f2=%5.2f score=%f invalid=%d\n",idx1,idx2,f1,f2,error,foundInvalid);
				}
			}

			if( minimumChanged ) {
				if (bestIndex > 0 && bestIndex < numSamples - 1) {
					localMinimum = bestError< errors[bestIndex - 1] && bestError < errors[bestIndex + 1];
				} else {
					localMinimum = false;
				}
			}
		}
		return bestError;
	}

	/**
	 * Given the focal lengths for the first two views compute homography H
	 * @param f1 view 1 focal length
	 * @param f2 view 2 focal length
	 * @param P2 projective camera matrix for view 2
	 * @param H (Output) homography
	 * @return true if successful
	 */
	boolean computeRectifyH( double f1 , double f2 , DMatrixRMaj P2, DMatrixRMaj H ) {

		estimatePlaneInf.setCamera1(f1,f1,0,0,0);
		estimatePlaneInf.setCamera2(f2,f2,0,0,0);

		if( !estimatePlaneInf.estimatePlaneAtInfinity(P2,planeInf) )
			return false;

		// TODO add a cost for distance from nominal and scale other cost by focal length fx for each view
//		RefineDualQuadraticConstraint refine = new RefineDualQuadraticConstraint();
//		refine.setZeroSkew(true);
//		refine.setAspectRatio(true);
//		refine.setZeroPrinciplePoint(true);
//		refine.setKnownIntrinsic1(true);
//		refine.setFixedCamera(false);
//
//		CameraPinhole intrinsic = new CameraPinhole(f1,f1,0,0,0,0,0);
//		if( !refine.refine(normalizedP.toList(),intrinsic,planeInf))
//			return false;

		K1.zero();
		K1.set(0,0,f1);
		K1.set(1,1,f1);
		K1.set(2,2,1);
		MultiViewOps.createProjectiveToMetric(K1,planeInf.x,planeInf.y,planeInf.z,1,H);

		return true;
	}

	double computeError() {

		MultiViewOps.projectiveToMetric(P1,H,view_1_to_1,K);
		PerspectiveOps.matrixToPinhole(K,0,0,intrinsic1);
		MultiViewOps.projectiveToMetric(P2,H,view_1_to_2,K);
		PerspectiveOps.matrixToPinhole(K,0,0,intrinsic2);
		MultiViewOps.projectiveToMetric(P3,H,view_1_to_3,K);
		PerspectiveOps.matrixToPinhole(K,0,0,intrinsic3);

//		System.out.println("cam2 focal fx="+intrinsic2.fx+" fy="+intrinsic2.fx);

//		view_1_to_1.print();

		double scale = view_1_to_2.T.norm();
		view_1_to_2.T.divide(scale);
		view_1_to_3.T.divide(scale);

		Triangulate2ViewsMetric triangulate = FactoryMultiView.triangulate2ViewMetric(null);

		matchesNorm.resize(matchesPixels.size());
		Point3D_F64 pointIn1 = new Point3D_F64();
		Point3D_F64 Xcam = new Point3D_F64();

		Point2D_F64 pixel = new Point2D_F64();

		Point3D_F64 Xcam0 = new Point3D_F64();


		AssociatedTriple an = new AssociatedTriple();

		foundInvalid = 0;
		double error = 0;
		double error3D = 0;

		for (int i = 0; i < matchesPixels.size(); i++) {
			AssociatedTriple ap = matchesPixels.get(i);

			PerspectiveOps.convertPixelToNorm(intrinsic1,ap.p1.x,ap.p1.y,an.p1);
			PerspectiveOps.convertPixelToNorm(intrinsic2,ap.p2.x,ap.p2.y,an.p2);

			triangulate.triangulate(an.p1,an.p2,view_1_to_2,pointIn1);
			SePointOps_F64.transform(view_1_to_3,pointIn1,Xcam);

			Xcam0.set(Xcam);

			if( Xcam.z < 0)
				foundInvalid++;

			PerspectiveOps.renderPixel(intrinsic3,Xcam,pixel);
			error += pixel.distance2(ap.p3);

			PerspectiveOps.convertPixelToNorm(intrinsic3,ap.p3.x,ap.p3.y,an.p3);
			triangulate.triangulate(an.p1,an.p3,view_1_to_3,pointIn1);
			SePointOps_F64.transform(view_1_to_2,pointIn1,Xcam);
			PerspectiveOps.renderPixel(intrinsic2,Xcam,pixel);
			error += pixel.distance2(ap.p2);

			error3D += Xcam.distance(Xcam0);

			if( Xcam.z < 0)
				foundInvalid++;
		}

//		System.out.println("     "+error3D);
		return error;
	}

	@Override
	public void setVerbose(@Nullable PrintStream out, @Nullable Set<String> configuration) {
		this.verbose = out;
	}
}
