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

import boofcv.abst.geo.Estimate1ofTrifocalTensor;
import boofcv.alg.geo.MultiViewOps;
import boofcv.alg.geo.PerspectiveOps;
import boofcv.struct.geo.AssociatedTriple;
import boofcv.struct.geo.TrifocalTensor;
import org.ddogleg.fitting.modelset.ModelGenerator;
import org.ejml.data.DMatrixRMaj;

import java.util.List;

/**
 * Computes a calibrating homography from a set of image triplets by estimating the focus by scoring candidates
 * with reprojection residuals.
 *
 * NOTE:
 * <ol>
 *     <li>It's assumed that pixel coordinates are centered around the principle point, i.e. (cx, cy) = (0,0)</li>
 *     <li>Skew is assumed to be zero.</li>
 * </ol>
 *
 * @see TrifocalBruteForceSelfCalibration
 *
 * @author Peter Abeles
 */
public class GenerateMetricCameraTripleBruteForceFocus implements ModelGenerator<MetricCameraTriple, AssociatedTriple> {

	// estimates the trifocal tensor
	Estimate1ofTrifocalTensor trifocal;
	// performs projective to metric self calibration
	TrifocalBruteForceSelfCalibration alg;

	//--------------- Internal Work Space
	TrifocalTensor tensor = new TrifocalTensor();
	DMatrixRMaj K = new DMatrixRMaj(3,3);

	public GenerateMetricCameraTripleBruteForceFocus(Estimate1ofTrifocalTensor trifocal, TrifocalBruteForceSelfCalibration alg) {
		this.trifocal = trifocal;
		this.alg = alg;
	}

	protected GenerateMetricCameraTripleBruteForceFocus() {}

	@Override
	public boolean generate(List<AssociatedTriple> dataSet, MetricCameraTriple result) {
		// Compute the trifocal tensor
		if( !trifocal.process(dataSet,tensor) )
			return false;

		// Projective to Metric calibration
		if( !alg.process(tensor,dataSet) )
			return false;
		DMatrixRMaj H = alg.rectifyingHomography;

		// copy and convert the results into the output format
		result.view1.fsetK(alg.focalLengthA,alg.focalLengthA,0,0,0,-1,-1);
		result.view2.fsetK(alg.focalLengthB,alg.focalLengthB,0,0,0,-1,-1);
		if( !MultiViewOps.projectiveToMetric(alg.getCameraMatrix2(),H,result.view_1_to_2,K) )
			return false;
		if( !MultiViewOps.projectiveToMetric(alg.getCameraMatrix3(),H,result.view_1_to_3,K) )
			return false;
		PerspectiveOps.matrixToPinhole(K,-1,-1,result.view3);

		return true;
	}

	@Override
	public int getMinimumPoints() {
		return trifocal.getMinimumPoints();
	}
}
