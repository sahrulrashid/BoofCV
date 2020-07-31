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
import org.ejml.UtilEjml;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * @author Peter Abeles
 */
class TestDistanceMetricTripleReprojection23 extends CommonThreeViewSelfCalibration {
	/**
	 * Give it perfect input and see if it computes the expected values
	 */
	@Test
	void perfect() {
		standardScene();
		simulateScene(0);

		var alg = new DistanceMetricTripleReprojection23();

		var model = new MetricCameraTriple();
		model.view1.set(cameraA);
		model.view2.set(cameraB);
		model.view3.set(cameraC);
		model.view_1_to_2.set(truthView_1_to_i(1));
		model.view_1_to_3.set(truthView_1_to_i(2));

		alg.setModel(model);

		for( AssociatedTriple a : observations ) {
			assertEquals(0.0, alg.computeDistance(a) , UtilEjml.TEST_F64);
		}

		var set = observations.subList(4,11);
		var distances = new double[ set.size() ];
		alg.computeDistance(set,distances);
		for (int i = 0; i < distances.length; i++) {
			assertEquals(0.0, distances[i], UtilEjml.TEST_F64 );
		}
	}

	/**
	 * Checks to see if distance() and distances() return the same value and the the error is larger than 0
	 */
	@Test
	void noisy() {
		standardScene();
		simulateScene(0);

		var alg = new DistanceMetricTripleReprojection23();

		var model = new MetricCameraTriple();
		model.view1.set(cameraA);
		model.view2.set(cameraB);
		model.view3.set(cameraC);
		model.view_1_to_2.set(truthView_1_to_i(1));
		model.view_1_to_3.set(truthView_1_to_i(2));

		model.view3.fx += 40; // this will mess things up a bit

		alg.setModel(model);

		var set = observations.subList(0,20);
		var distances = new double[ set.size() ];
		alg.computeDistance(set,distances);
		for (int i = 0; i < distances.length; i++) {
			assertTrue(distances[i] > 0.1 );
			assertEquals(distances[i], alg.computeDistance(set.get(i)), UtilEjml.TEST_F64 );
		}
	}
}