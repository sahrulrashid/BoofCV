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

import boofcv.alg.geo.PerspectiveOps;
import boofcv.struct.calib.CameraPinhole;
import boofcv.struct.geo.AssociatedTriple;
import boofcv.testing.BoofTesting;
import georegression.geometry.UtilPoint3D_F64;
import georegression.struct.point.Point3D_F64;
import georegression.struct.se.Se3_F64;
import georegression.struct.se.SpecialEuclideanOps_F64;
import org.ejml.data.DMatrixRMaj;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * @author Peter Abeles
 */
public class TestSelfCalibrationGuessAndCheckFocusV2 {
	Random rand = BoofTesting.createRandom(6);
	double focus = 600;

	List<Se3_F64> list_world_to_cameras = new ArrayList<>();
	List<Point3D_F64> cloud = new ArrayList<>();
	List<AssociatedTriple> observations = new ArrayList<>();
	List<DMatrixRMaj> projective = new ArrayList<>();

	private void simulateScene() {
		list_world_to_cameras.add( SpecialEuclideanOps_F64.eulerXyz(0,0,-2,0.03,-0.06,0.06,null).invert(null));
		list_world_to_cameras.add( SpecialEuclideanOps_F64.eulerXyz(0.5,0,-2.5,0.01,-0.04,0.00,null).invert(null));
		list_world_to_cameras.add( SpecialEuclideanOps_F64.eulerXyz(-0.3,0.1,-2,-0.03,0.02,0.02,null).invert(null));

		cloud = UtilPoint3D_F64.random(-1,-1,100,rand);

		var camera = new CameraPinhole(focus,focus,0,0,0,800,600);

		DMatrixRMaj K = PerspectiveOps.pinholeToMatrix(camera,(DMatrixRMaj)null);

		for( Se3_F64 world_to_camera : list_world_to_cameras ) {
			projective.add(PerspectiveOps.createCameraMatrix(world_to_camera.R,world_to_camera.T,K,null));
		}

		for( Point3D_F64 X : cloud ) {
			AssociatedTriple a = new AssociatedTriple();
			PerspectiveOps.renderPixel(list_world_to_cameras.get(0),camera,X,a.p1);
			PerspectiveOps.renderPixel(list_world_to_cameras.get(1),camera,X,a.p2);
			PerspectiveOps.renderPixel(list_world_to_cameras.get(2),camera,X,a.p3);

			double sigma = 0.0;
			a.p1.x += rand.nextGaussian()*sigma;
			a.p1.y += rand.nextGaussian()*sigma;
			a.p2.x += rand.nextGaussian()*sigma;
			a.p2.y += rand.nextGaussian()*sigma;
			a.p3.x += rand.nextGaussian()*sigma;
			a.p3.y += rand.nextGaussian()*sigma;

			observations.add(a);
		}
	}


	@Test
	public void perfect_data_oneK() {
		simulateScene();

		var alg = new SelfCalibrationGuessAndCheckFocusV2();
		alg.setSameFocus(true);
		alg.setVerbose(System.out,null);

		assertTrue(alg.process(projective.get(0),projective.get(1),projective.get(2),observations));

		System.out.println("Best focus: "+alg.getBestFocus1());
		System.out.println("Best focus: "+alg.getBestFocus2());

	}
}