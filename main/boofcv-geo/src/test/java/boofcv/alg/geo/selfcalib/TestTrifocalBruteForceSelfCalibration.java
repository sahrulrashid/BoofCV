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

import boofcv.struct.calib.CameraPinhole;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author Peter Abeles
 */
public class TestTrifocalBruteForceSelfCalibration extends CommonThreeViewSelfCalibration {

	/**
	 * fixed focus for all the cameras
	 */
	@Test
	public void perfect_fixedFocus() {
		standardScene();
		var camera = new CameraPinhole(700,700,0.0,0,0,800,600);
		setCameras(camera,camera,camera);
		simulateScene(0);

		var alg = new TrifocalBruteForceSelfCalibration();
		alg.fixedFocus = true;
		alg.configure(200,2000);
		alg.process(tensor,observations);

		assertFalse(alg.isLimit);
		assertEquals(camera.fx, alg.focalLengthA, 25);
		assertEquals(camera.fx, alg.focalLengthB, 25);
	}

	/**
	 * See if it can estimate two different camera models
	 */
	@Test
	public void perfect_two_cameras() {
		standardScene();
		var camera1 = new CameraPinhole(700,700,0.0,0,0,800,600);
		var camera2 = new CameraPinhole(450,450,0.0,0,0,800,600);

		setCameras(camera1,camera2,camera2);
		simulateScene(0);

		var alg = new TrifocalBruteForceSelfCalibration();
		alg.fixedFocus = false;
		alg.configure(200,2000);
		alg.process(tensor,observations);

		assertFalse(alg.isLimit);
		assertEquals(camera1.fx, alg.focalLengthA, 25);
		assertEquals(camera2.fx, alg.focalLengthB, 25);
	}

	/**
	 * See if it blows up if noise is added
	 */
	@Test
	public void noisy_two_cameras() {
		standardScene();
		var camera1 = new CameraPinhole(700,700,0.0,0,0,800,600);
		var camera2 = new CameraPinhole(450,450,0.0,0,0,800,600);

		setCameras(camera1,camera2,camera2);
		simulateScene(0.25);

		var alg = new TrifocalBruteForceSelfCalibration();
		alg.fixedFocus = false;
		alg.configure(200,2000);
		alg.process(tensor,observations);

		assertFalse(alg.isLimit);
		assertEquals(camera1.fx, alg.focalLengthA, 25);
		assertEquals(camera2.fx, alg.focalLengthB, 25);
	}

	/**
	 * See if the hit limit flag actually works
	 */
	@Test
	public void hit_limit() {
		standardScene();
		var camera = new CameraPinhole(1500,1500,0.0,0,0,800,600);
		setCameras(camera,camera,camera);
		simulateScene(0);

		var alg = new TrifocalBruteForceSelfCalibration();
		alg.fixedFocus = true;
		alg.configure(200,1000);
		alg.process(tensor,observations);

		// true value of focal length is greater than the range it will test. It should git the limit
		assertTrue(alg.isLimit);
		assertEquals(1000, alg.focalLengthA, 25);
		assertEquals(1000, alg.focalLengthB, 25);
	}
}