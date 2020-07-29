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
import boofcv.alg.geo.DecomposeEssential;
import boofcv.alg.geo.MultiViewOps;
import boofcv.alg.geo.PerspectiveOps;
import boofcv.alg.geo.trifocal.TrifocalExtractGeometries;
import boofcv.factory.geo.FactoryMultiView;
import boofcv.struct.calib.CameraPinhole;
import boofcv.struct.geo.AssociatedTriple;
import boofcv.struct.geo.TrifocalTensor;
import georegression.struct.point.Point2D_F64;
import georegression.struct.point.Point3D_F64;
import georegression.struct.point.Vector3D_F64;
import georegression.struct.se.Se3_F64;
import georegression.transform.se.SePointOps_F64;
import lombok.Getter;
import org.ddogleg.struct.FastQueue;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.linsol.LinearSolver;

import java.util.List;

/**
 * Estimates the calibrating/rectifying homography when given a trifocal tensor and two calibration matrices for
 * the first two views. Observations are used to select the best hypothesis out of the four possible camera motions.
 *
 * Procedure:
 * <ol>
 *     <li>Get fundamental and camera matrices from trifocal tensor</li>
 *     <li>Use given calibration matrices to compute essential matrix</li>
 *     <li>Decompose essential matrix to get 4 possible motions from view 1 to view 2</li>
 *     <li>Use reprojection error and visibility constraints to select best hypothesis</li>
 * </ol>
 *
 * Reprojection error is computed by triangulating each point in view-1 using views-1 and view-2. This is then
 * switched to view-3's reference frame and the reprojection error found there. A similar process is repeated using
 * triangulation from view-1 and view-3. In each view it's checked if the feature appears behind the camera and
 * increments the invalid counter if it does.
 *
 * When selecting a hypothesis the hypothesis with the most points appearing in front of call cameras is given priority
 * over lower reprojection error.
 *
 * When applied to view 2, the found translation should have a norm(T) = 1.
 *
 * <ol>
 * <li> P. Abeles, "BoofCV Technical Report: Automatic Camera Calibration" 2020-1 </li>
 * </ol>
 *
 * @author Peter Abeles
 */
public class TrifocalToCalibratingHomography {

	/** used to triangulate feature locations when checking a solution */
	public Triangulate2ViewsMetric triangulate = FactoryMultiView.triangulate2ViewMetric(null);

	// used to decompose the trifocal tensor
	public final TrifocalExtractGeometries trifocalGeo = new TrifocalExtractGeometries();
	// Decomposes the essential matrix
	public final DecomposeEssential decomposeEssential = new DecomposeEssential();

	// Camera matrices extracted from the trifocal tensor.
	public final DMatrixRMaj P2 = new DMatrixRMaj(3,4);
	public final DMatrixRMaj P3 = new DMatrixRMaj(3,4);

	// Extracted fundamental matrices from trifocal tensor
	public final DMatrixRMaj F21 = new DMatrixRMaj(3,3);
	public final DMatrixRMaj F31 = new DMatrixRMaj(3,3);

	// Essential matrix for the first two views
	public final DMatrixRMaj E21 = new DMatrixRMaj(3,3);

	//--------------------------------------------------------------------------------
	// Output Data Structures
	/** List of all hopotheses for calibrating homography */
	public final FastQueue<DMatrixRMaj> hypothesesH = new FastQueue<>(()->new DMatrixRMaj(4,4));
	/** Which hypothesis was selected as the best. Call {@link #getCalibrationHomography()} as an alternative */
	public @Getter int bestSolutionIdx;
	/** The number of invalid observations that appeared behind the camera in the best hypothesis */
	public int bestInvalid = Integer.MAX_VALUE;
	/** The sum of reprojection error for the best hypothesis */
	public double bestError = Double.MAX_VALUE;

	//------------------------------------------------------------------------------
	// work space variables
	private final DMatrixRMaj A = new DMatrixRMaj(3,3);
	private final Vector3D_F64 a = new Vector3D_F64();
	private final DMatrixRMaj AK1 = new DMatrixRMaj(3,3);
	private final DMatrixRMaj KiR = new DMatrixRMaj(3,3);

	// Given and estimated intrinsic calibration
	private final CameraPinhole intrinsic1 = new CameraPinhole();
	private final CameraPinhole intrinsic2 = new CameraPinhole();
	private final CameraPinhole intrinsic3 = new CameraPinhole();

	// motion from camera views
	private final Se3_F64 view_1_to_2 = new Se3_F64();
	private final Se3_F64 view_1_to_3 = new Se3_F64();

	private final DMatrixRMaj K3 = new DMatrixRMaj(3,3);
	private final DMatrixRMaj calibratingH = new DMatrixRMaj(4,4);

	// used to keep track of the number of invalid observations in a hypothesis
	private int foundInvalid;

	// location of 3D feature in view 1
	private final Point3D_F64 pointIn1 = new Point3D_F64();
	// location of 3D feature in the current view being considered
	private final Point3D_F64 Xcam = new Point3D_F64();
	// Projected location of feature in camera
	private final Point2D_F64 pixel = new Point2D_F64();
	// storage for normalized image coordinates
	private final AssociatedTriple an = new AssociatedTriple();

	// Linear solver
	private final LinearSolver<DMatrixRMaj,DMatrixRMaj> linear = LinearSolverFactory_DDRM.leastSquares(9,3);
	private final DMatrixRMaj matA = new DMatrixRMaj(9,3);
	private final DMatrixRMaj matX = new DMatrixRMaj(9,1);
	private final DMatrixRMaj matB = new DMatrixRMaj(9,1);

	/**
	 * Sets the trifocal tensor which is used to seed the self calibration process
	 *
	 * @param T (Input) trifocal tensor for the tree views
	 */
	public void setTrifocalTensor( TrifocalTensor T ) {
		trifocalGeo.setTensor(T);
		trifocalGeo.extractFundmental(F21,F31);
		trifocalGeo.extractCamera(P2,P3);
	}

	/**
	 * Estimate the calibrating homography with the given assumptions about the intrinsic calibration matrices
	 * for the first two of three views.
	 *
	 * @param K1 (input) known intrinsic camera calibration matrix for view-1
	 * @param K2 (input) known intrinsic camera calibration matrix for view-2
	 * @param observations (input) observations for all three views. Used to select best solution
	 * @return true if it could find a solution. Failure is a rare condition which requires noise free data.
	 */
	public boolean process(DMatrixRMaj K1 , DMatrixRMaj K2 , List<AssociatedTriple> observations )
	{
		// TODO try to improve numerics by reducing the scale of K1 and K2 to 0 to 1.0 for diagonal elements
		//      then undo it

		bestSolutionIdx = -1;
		// Using the provided calibration matrices, extract potential camera motions
		MultiViewOps.fundamentalToEssential(F21,K1,K2,E21);
		decomposeEssential.decompose(E21);

		// Use these camera motions to guess different calibrating homographies
		List<Se3_F64> list_view_1_to_2 = decomposeEssential.getSolutions();
		computeHypothesesForH(K1, K2, list_view_1_to_2);
		// DESIGN NOTE: Could swap the role of view-2 and view-3 if view-2 is pathological

		// Select the best hypothesis
		bestInvalid = Integer.MAX_VALUE;
		bestError = Double.MAX_VALUE;
		for (int motionIdx = 0; motionIdx < hypothesesH.size; motionIdx++) {

			// computes the reprojection error, valid projections, and fixes sign/scale of H
			double error = computeScore(list_view_1_to_2.get(motionIdx),hypothesesH.get(motionIdx),K1,K2,observations);
			if( error == Double.MAX_VALUE )
				continue;

			boolean better = false;
			if( foundInvalid < bestInvalid ) {
				better = true;
			} else if( foundInvalid == bestInvalid && error < bestError ) {
				better = true;
			}

			if( better ) {
				bestInvalid = foundInvalid;
				bestError = error;
				bestSolutionIdx = motionIdx;
			}
//			System.out.println(motionIdx+" invalid="+foundInvalid+" score="+score);
		}
		return bestSolutionIdx >= 0;
	}

	/**
	 * Returns the found calibration/rectifying homography.
	 */
	public DMatrixRMaj getCalibrationHomography() {
		return hypothesesH.get(bestSolutionIdx);
	}

	/**
	 * Go through all the found camera motions and generate a hypothesis for each one. Care is taken to compute
	 * the hypothesis in a numerically stable way. The left and right hand side of the equation (see in code comments)
	 * are only equal up to a scale factor. So first the scale factor is found by computing it several times and
	 * picking the one with the largest denominator to avoid numerical issues. Once the scale factor is known then
	 * a linear system is created that can be easily solved for.
	 *
	 * Technically the solution is found when finding the scale factor, but only a single equation for each unknown
	 * is used there. Once the scale factor is known then all the variables can be used resulting in a more stable
	 * solution.
	 */
	void computeHypothesesForH(DMatrixRMaj K1, DMatrixRMaj K2, List<Se3_F64> list_view_1_to_2) {
		// P2*H ~= [A,a]*H = [A,a]*[K1 0;v',1]
		// AK ~= A*K1
		PerspectiveOps.projectionSplit(P2,A,a);
		CommonOps_DDRM.mult(A,K1, AK1);

		CommonOps_DDRM.insert(K1,calibratingH,0,0);
		calibratingH.set(3,3,1);

		hypothesesH.reset();

		for( int motionIdx = 0; motionIdx < list_view_1_to_2.size(); motionIdx++ ) {
			view_1_to_2.set(list_view_1_to_2.get(motionIdx));

			// K2*[R,T] = [K2*R, K2*T] = P2*H
			// KR = K2*R
			CommonOps_DDRM.mult(K2,view_1_to_2.R, KiR);

			// Find the scale factor between AK1 and AKiR. Brute force through all possible combinations and select
			// the one which is least prone to numerical instability due to a small denominator
			double bestBottom = 0;
			double bestScale = 0.0;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					for (int k = 0; k < 3; k++) {
						if( i == k )
							continue;
						double top = AK1.get(i,j)*a.getIdx(k) - AK1.get(k,j)*a.getIdx(i);
						double bottom = a.getIdx(k)*KiR.get(i,j) - a.getIdx(i)*KiR.get(k,j);
						double scale = top/bottom;

						if( Math.abs(bottom) > bestBottom ) {
							bestBottom = Math.abs(bottom);
							bestScale = scale;
						}
//						System.out.println("i="+i+" j="+j+" k="+k+" scale["+j+"] = "+scale+"  bottom = "+bottom);
					}
				}
			}

			// Construct a linear system and solve for the 3 unknowns in v. A linear system is used rather than the
			// commented out algebraic solution de to the possibility of it blowing up if sum of "a" is zero
			for (int j = 0, row=0; j < 3; j++) {
				for (int i = 0; i < 3; i++, row++) {
					matA.set(row,j, a.getIdx(i));
					matB.set(row,0, KiR.get(i,j)*bestScale-AK1.get(i,j));
				}
			}

			if( !linear.setA(matA) ) {
				continue;
			}
			linear.solve(matB,matX);
			for (int i = 0; i < 3; i++) {
				calibratingH.set(3, i, matX.get(i));
			}

//			// algebraic solution, but has at least one known issue
//			for (int j = 0; j < 3; j++) {
//				double sumA = 0;
//				double sumK = 0;
//				for (int i = 0; i < 3; i++) {
//					sumA += AK1.get(i, j);
//					sumK += KiR.get(i, j);
//				}
//				double v_j = (sumK * bestScale - sumA) / (a.x + a.y + a.z); // NOTE: Degenerate geometry here of sum is zero?
//				calibratingH.set(3, j, v_j);
//				System.out.println("v[" + j + "] = " + v_j);
//			}
			// DESIGN NOTE:
			// Could Lagrange multipliers be used here where KiR is known to have zeros?

			hypothesesH.grow().set(calibratingH);
		}
	}

	/**
	 * Scores the hypothesis using reprojection error and by assuming a feature has to appear in front of the a camera
	 *
	 * @param view_1_to_2e (Input) camera motion returned by essential matrix
	 * @param H (Input, Output) Calibrating homography. Modifies H(3,3) to set the scale to something reasonable and
	 *          for direction
	 * @param K1 (Input) Calibration matrix for view 1
	 * @param K2 (Input) Calibration matrix for view 1
	 * @param observations (Input) observations from all 3 cameras
	 * @return sum of reprojection error
	 */
	private double computeScore(Se3_F64 view_1_to_2e, DMatrixRMaj H, DMatrixRMaj K1 , DMatrixRMaj K2,
								List<AssociatedTriple> observations) {
		PerspectiveOps.matrixToPinhole(K1,0,0,intrinsic1);
		PerspectiveOps.matrixToPinhole(K2,0,0,intrinsic2);

//		MultiViewOps.projectiveToMetricKnownK(P2,H,K2,view_1_to_2); <-- NOT NOT USE THIS VARIANT! Loses common scale
		if( !MultiViewOps.projectiveToMetric(P2,H,view_1_to_2,K3) ) // K3 is used as a dummy output since K2 is known
			return Double.MAX_VALUE;
		if( !MultiViewOps.projectiveToMetric(P3,H,view_1_to_3,K3) )
			return Double.MAX_VALUE;
		PerspectiveOps.matrixToPinhole(K3,0,0,intrinsic3);

		// make sure the camera motion from H has a norm of 1
		double scale = view_1_to_2.T.norm();
		view_1_to_2.T.divide(scale);
		// the sign can get messed up by H
		if( view_1_to_2e.T.distance(view_1_to_2.T) > 1.0 ) {
			scale *= -1;
		}
		view_1_to_2.set(view_1_to_2e);
		view_1_to_3.T.divide(scale);

		// Fix the rectifying homography so that the translation vector has a scale of 1 an is pointing in the right
		// direction
		H.set(3,3,1.0/scale);

		// count the number of times it appears behind a camera
		foundInvalid = 0;
		// storage for reprojection error
		double error = 0;
		// DESIGN NOTE:
		// On a small set of images, scoring with the 50% error produced nearly the same results. In one image
		// it did result in a significant improvement in performance.

		for (int i = 0; i < observations.size(); i++) {
			AssociatedTriple ap = observations.get(i);

			PerspectiveOps.convertPixelToNorm(intrinsic1,ap.p1.x,ap.p1.y,an.p1);
			PerspectiveOps.convertPixelToNorm(intrinsic2,ap.p2.x,ap.p2.y,an.p2);

			triangulate.triangulate(an.p1,an.p2,view_1_to_2,pointIn1);
			if( pointIn1.z < 0)
				foundInvalid++;

			SePointOps_F64.transform(view_1_to_3,pointIn1,Xcam);
			PerspectiveOps.renderPixel(intrinsic3,Xcam,pixel);
			error += pixel.distance2(ap.p3);
			if( Xcam.z < 0)
				foundInvalid++;

			PerspectiveOps.convertPixelToNorm(intrinsic3,ap.p3.x,ap.p3.y,an.p3);
			triangulate.triangulate(an.p1,an.p3,view_1_to_3,pointIn1);
			SePointOps_F64.transform(view_1_to_2,pointIn1,Xcam);
			PerspectiveOps.renderPixel(intrinsic2,Xcam,pixel);
			error += pixel.distance2(ap.p2);
			if( Xcam.z < 0)
				foundInvalid++;
		}

		return error;
	}
}
