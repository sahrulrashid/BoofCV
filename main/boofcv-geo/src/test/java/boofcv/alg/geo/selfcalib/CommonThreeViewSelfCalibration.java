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

import boofcv.alg.geo.MultiViewOps;
import boofcv.alg.geo.PerspectiveOps;
import boofcv.misc.BoofMiscOps;
import boofcv.struct.calib.CameraPinhole;
import boofcv.struct.geo.AssociatedTriple;
import boofcv.struct.geo.TrifocalTensor;
import boofcv.testing.BoofTesting;
import georegression.geometry.UtilPoint3D_F64;
import georegression.struct.point.Point3D_F64;
import georegression.struct.se.Se3_F64;
import org.ejml.data.DMatrixRMaj;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static georegression.struct.se.SpecialEuclideanOps_F64.eulerXyz;

/**
 * @author Peter Abeles
 */
public class CommonThreeViewSelfCalibration {
	protected Random rand = BoofTesting.createRandom(6);
	protected CameraPinhole cameraA;
	protected CameraPinhole cameraB;
	protected CameraPinhole cameraC;

	protected int numFeatures = 100;

	protected List<Se3_F64> list_world_to_cameras;
	protected List<Point3D_F64> cloud;
	protected List<AssociatedTriple> observations;
	protected List<DMatrixRMaj> projective;

	protected TrifocalTensor tensor = new TrifocalTensor();

	protected void standardScene() {
		cameraA = new CameraPinhole(600,600,0.1,400,410,800,600);
		cameraB = cameraC = cameraA;
		list_world_to_cameras = new ArrayList<>();
		list_world_to_cameras.add( eulerXyz(0,0,-2,0,0,0.0,null).invert(null));
		list_world_to_cameras.add( eulerXyz(1,0.1,-2,-0.1,0,0.05,null).invert(null));
		list_world_to_cameras.add( eulerXyz(0.1,1,-2,0,0.05,0.0,null).invert(null));
	}

	/**
	 * Returns the SE3 transform from view 1 to view 'i'
	 */
	protected Se3_F64 truthView_1_to_i( int i ) {
		Se3_F64 world_to_1 = list_world_to_cameras.get(0);
		Se3_F64 world_to_i = list_world_to_cameras.get(i);

		return world_to_1.invert(null).concat(world_to_i,null);
	}

	protected void setCameras( CameraPinhole a , CameraPinhole b, CameraPinhole c ) {
		this.cameraA = a;
		this.cameraB = b;
		this.cameraC = c;
	}

	protected void simulateScene( double noiseSigma ) {
		List<CameraPinhole> cameras = BoofMiscOps.asList(cameraA,cameraB,cameraC);
		cloud = new ArrayList<>();
		observations = new ArrayList<>();
		projective = new ArrayList<>();

		cloud = UtilPoint3D_F64.random(-1,1,numFeatures,rand);

		BoofMiscOps.forIdx(list_world_to_cameras,(idx,world_to_camera)->{
			DMatrixRMaj K = PerspectiveOps.pinholeToMatrix(cameras.get(idx),(DMatrixRMaj)null);
			projective.add(PerspectiveOps.createCameraMatrix(world_to_camera.R,world_to_camera.T,K,null));
		});

		for( Point3D_F64 X : cloud ) {
			AssociatedTriple a = new AssociatedTriple();
			PerspectiveOps.renderPixel(list_world_to_cameras.get(0), cameraA,X,a.p1);
			PerspectiveOps.renderPixel(list_world_to_cameras.get(1), cameraB,X,a.p2);
			PerspectiveOps.renderPixel(list_world_to_cameras.get(2), cameraC,X,a.p3);

			a.p1.x += rand.nextGaussian()*noiseSigma;
			a.p1.y += rand.nextGaussian()*noiseSigma;
			a.p2.x += rand.nextGaussian()*noiseSigma;
			a.p2.y += rand.nextGaussian()*noiseSigma;
			a.p3.x += rand.nextGaussian()*noiseSigma;
			a.p3.y += rand.nextGaussian()*noiseSigma;

			observations.add(a);
		}

		MultiViewOps.createTrifocal(projective.get(0),projective.get(1),projective.get(2),tensor);
	}
}
