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

import boofcv.struct.geo.AssociatedTriple;
import boofcv.struct.geo.TrifocalTensor;
import lombok.Getter;
import org.ddogleg.struct.VerbosePrint;
import org.ejml.data.DMatrixRMaj;

import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.List;
import java.util.Set;

import static boofcv.misc.BoofMiscOps.assertBoof;

/**
 * <p>
 * Brute force sampling approach to perform self calibration of a partially calibrated image. The focal length
 * for the first and second images are selected by sampling a grid of values and selecting the hypothesis with
 * the smallest reprojection error. It can be configured to assume the first two images have the same
 * focal length or not.
 * </p>
 *
 * Assumptions: zero skew, zero principle point, and no lens distortion. There are no false positives in the input set.
 *
 * Steps:
 * <ol>
 *     <li>Decompose trifocal tensor into camera matrices and fundamental matrix</li>
 *     <li>Create a list of different focal length hypothesis</li>
 *     <li>Use a given hypothesis to compute the Essential matrix</li>
 *     <li>Decompose essential matrix and get 4 extrinsic hypotheses</li>
 *     <li>Select best hypothesis and save result. Repeat for all focal lengths</li>
 * </ol>
 * The list of hypothetical focal lengths is generated using a log scale causing it to generate more hypotheses on
 * the lower end. The camera matrix for view-1 is an 3x4 identity matrix.
 *
 * The output comes in various forms:
 * <ul>
 *     <li>Select focal lengths for camera</li>
 *     <li>Rectifying homography</li>
 * </ul>
 *
 * <ol>
 * <li> P. Abeles, "BoofCV Technical Report: Automatic Camera Calibration" 2020-1 </li>
 * </ol>
 *
 * @see TrifocalToCalibratingHomography
 *
 * @author Peter Abeles
 */
public class TrifocalBruteForceSelfCalibration implements VerbosePrint {

	/** Generates and scores a hypothesis given two intrinsic camera matrices */
	public @Getter final TrifocalToCalibratingHomography calibrator = new TrifocalToCalibratingHomography();

	/** Range of values it will sample */
	public @Getter double sampleMin,sampleMax;
	/** Number of values it will sample */
	public @Getter int numberOfSamples=50;

	/** if true the focus is assumed to be the same for the first two images*/
	public @Getter boolean fixedFocus =false;

	/** The selected focal length for the first image */
	public @Getter double focalLengthA;
	/** The selected focal length for the second image */
	public @Getter double focalLengthB;

	/** The selected rectifying homography */
	public @Getter final DMatrixRMaj rectifyingHomography = new DMatrixRMaj(4,4);

	/** If true that indicates that the selected focal length was at the upper or lower limit. This can indicate a fault */
	public @Getter boolean isLimit;

	// If not null then verbose information is printed
	private PrintStream verbose;

	/**
	 * Specifies the range of focal lengths it will evaluate
	 * @param sampleMin The minimum allowed focal length
	 * @param sampleMax Tha maximum allowed focal length
	 */
	public void configure( double sampleMin , double sampleMax ) {
		this.sampleMin = sampleMin;
		this.sampleMax = sampleMax;
	}

	/**
	 * Selects the best focal length(s) given the trifocal tensor and observations
	 * @param T (Input) trifocal tensor
	 * @param observations (Input) Observation for all three views. Highly recommend that RANSAC or similar is used to
	 *                     remove false positives first.
	 * @return true if successful
	 */
	public boolean process(TrifocalTensor T, List<AssociatedTriple> observations ) {
		// sanity check configurations
		assertBoof(sampleMin!=0 && sampleMax !=0,"You must call configure");
		assertBoof(sampleMin < sampleMax && sampleMin > 0);
		assertBoof(observations.size()>0);
		assertBoof(numberOfSamples>0);

		// Pass in the trifocal tensor so that it can estimate self calibration
		calibrator.setTrifocalTensor(T);

		// coeffients for linear to log scale
		double logCoef = Math.log(sampleMax/sampleMin)/(numberOfSamples-1);

		DMatrixRMaj K1 = new DMatrixRMaj(3,3);
		K1.set(2,2,1);

		isLimit = false;

		if(fixedFocus) {
			searchFixedFocus(observations, logCoef, K1);
		} else {
			searchDynamicFocus(observations, logCoef, K1);
		}

		// DESIGN NOTE:
		// Could fit a 1-D or 2-D quadratic and get additional accuracy. Then compute the H at that value
		// In real data it seems that outliers drive the error more than a slightly incorrect focal length
		// so it's probably not worth the effort.

		return true;
	}

	/**
	 * Assumes that each camera can have an independent focal length value and searches a 2D grid
	 *
	 * @param observations observations of the features used to select beset result
	 * @param logCoef coeffient for log scale
	 * @param K1 intrinsic camera calibration matrix for view-1
	 */
	private void searchDynamicFocus(List<AssociatedTriple> observations, double logCoef, DMatrixRMaj K1) {
		double bestError = Double.MAX_VALUE;

		var K2 = new DMatrixRMaj(3,3);
		K2.set(2,2,1);

		for (int idxA = 0; idxA < numberOfSamples; idxA++) {
			double focalA = sampleMin * Math.exp(logCoef * idxA);
			K1.set(0, 0, focalA);
			K1.set(1, 1, focalA);

			for (int idxB = 0; idxB < numberOfSamples; idxB++) {
				double focalB = sampleMin * Math.exp(logCoef * idxB);

				K2.set(0, 0, focalB);
				K2.set(1, 1, focalB);

				calibrator.process(K1, K2, observations);

				double error = calibrator.bestError;
				if( verbose != null )
					verbose.printf("[%3d,%3d] f1=%5.2f f2=%5.2f error=%f invalid=%d\n",idxA,idxB,focalA,focalB,error,calibrator.bestInvalid);
				if( error < bestError ) {
					isLimit = idxA == 0 || idxA == numberOfSamples-1;
					isLimit |= idxB == 0 || idxB == numberOfSamples-1;
					bestError = error;
					focalLengthA = focalA;
					focalLengthB = focalB;
					rectifyingHomography.set(calibrator.getCalibrationHomography());
				}
			}
		}
	}

	/**
	 * Assumes that there is only one focal length value and searches for the optical value
	 *
	 * @param observations observations of the features used to select beset result
	 * @param logCoef coeffient for log scale
	 * @param K1 intrinsic camera calibration matrix for view-1
	 */
	private void searchFixedFocus(List<AssociatedTriple> observations, double logCoef, DMatrixRMaj K1) {
		double bestError = Double.MAX_VALUE;

		for (int idxA = 0; idxA < numberOfSamples; idxA++) {
			double focalA = sampleMin * Math.exp(logCoef * idxA);
			K1.set(0, 0, focalA);
			K1.set(1, 1, focalA);

			calibrator.process(K1, K1, observations);

			double error = calibrator.bestError;
			if( verbose != null )
				verbose.printf("[%3d] f=%5.2f error=%f invalid=%d\n", idxA, focalA, error, calibrator.bestInvalid);

			if( error < bestError ) {
				isLimit = idxA == 0 || idxA == numberOfSamples-1;
				bestError = error;
				focalLengthA = focalA;
				rectifyingHomography.set(calibrator.getCalibrationHomography());
			}
		}
		// Copy results to the other camera
		focalLengthB = focalLengthA;
	}

	/** Camera matrix for view-2 */
	public DMatrixRMaj getCameraMatrix2() {	return calibrator.P2; }

	/** Camera matrix for view-3 */
	public DMatrixRMaj getCameraMatrix3() {	return calibrator.P3; }

	@Override
	public void setVerbose(@Nullable PrintStream out, @Nullable Set<String> configuration) {
		this.verbose = out;
	}
}
