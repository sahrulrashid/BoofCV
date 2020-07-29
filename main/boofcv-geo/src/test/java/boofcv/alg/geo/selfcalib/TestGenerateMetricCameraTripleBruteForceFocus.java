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
import boofcv.factory.geo.FactoryMultiView;
import boofcv.struct.geo.AssociatedTriple;
import org.ddogleg.fitting.modelset.ModelGenerator;

/**
 * @author Peter Abeles
 */
class TestGenerateMetricCameraTripleBruteForceFocus extends CommonGenerateMetricCameraTripleChecks {
	@Override
	public ModelGenerator<MetricCameraTriple, AssociatedTriple> createGenerator() {

		Estimate1ofTrifocalTensor trifocal = FactoryMultiView.trifocal_1(null);
		var alg = new TrifocalBruteForceSelfCalibration();
		alg.fixedFocus = false;
//		alg.numberOfSamples = 200;
		alg.configure(300,1200); // narrower range
//		alg.configure(300,2500); <-- if the number of samples is increased it will pass with this, but is slow right now
//		alg.setVerbose(System.out,null);

		return new GenerateMetricCameraTripleBruteForceFocus(trifocal,alg);
	}
}