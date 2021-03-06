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

package boofcv.alg.sfm.d3;

import boofcv.abst.geo.TriangulateNViewsMetric;
import boofcv.abst.tracker.PointTrack;
import boofcv.alg.sfm.d3.structure.VisOdomBundleAdjustment;
import boofcv.alg.sfm.d3.structure.VisOdomBundleAdjustment.BFrame;
import boofcv.alg.sfm.d3.structure.VisOdomBundleAdjustment.BTrack;
import boofcv.factory.distort.LensDistortionFactory;
import boofcv.factory.geo.FactoryMultiView;
import boofcv.struct.calib.CameraPinholeBrown;
import georegression.struct.point.Point2D_F64;
import georegression.struct.point.Point3D_F64;
import georegression.struct.se.Se3_F64;
import org.ddogleg.struct.GrowQueue_I32;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * @author Peter Abeles
 */
class TestVisOdomBundlePnPBase {

	CameraPinholeBrown pinhole = new CameraPinholeBrown(400,400,0,500,500,1000,1000);

	@Test
	void reset() {
		var alg = new BundleBase();
		alg.first = false;
		alg.current_to_world.T.z = 10;
		alg.cameraModels.add( new VisOdomBundlePnPBase.CameraModel());

		alg.reset();

		assertTrue(alg.first);
		assertEquals(0,alg.current_to_world.T.z);
		assertEquals(0,alg.cameraModels.size());
	}

	/**
	 * Checks to see if it filters tracks based on observation count and being selected or not.
	 */
	@Test
	void triangulateNotSelectedBundleTracks_filter() {
		var mock = new MockTriangulate();
		var alg = new BundleBase();
		alg.scene.addCamera(pinhole);
		alg.triangulateN = mock;

		VisOdomBundlePnPBase.CameraModel cmodel = new VisOdomBundlePnPBase.CameraModel();
		cmodel.pixelToNorm = LensDistortionFactory.narrow(pinhole).undistort_F64(true,false);
		alg.cameraModels.add(cmodel);

		for (int fidx = 0; fidx < 3; fidx++) {
			alg.scene.addFrame(fidx);
		}

		for (int tidx = 0; tidx < 20; tidx++) {
			BTrack bt = alg.scene.addTrack(0,0,1,1.0);
			int N = tidx < 10 ? 2 : 3;
			for (int fidx = 0; fidx < N; fidx++) {
				alg.scene.addObservation(alg.scene.frames.get(fidx),bt,0,0);
			}
			bt.selected = tidx%2==0;
		}

		alg.triangulateNotSelectedBundleTracks();
		// only half will have enough observations and out of that half, half will be selected
		assertEquals(5,mock.count);
	}

	private static class MockTriangulate implements TriangulateNViewsMetric {

		public int count  = 0;

		@Override
		public boolean triangulate(List<Point2D_F64> observations, List<Se3_F64> worldToView, Point3D_F64 location) {
			assertEquals(3,observations.size());
			assertEquals(3,worldToView.size());
			count++;
			return true;
		}
	}

	@Test
	void dropFramesFromScene() {
		var alg = new BundleBase();
		alg.scene.addCamera(pinhole);

		for (int fidx = 0; fidx < 10; fidx++) {
			alg.scene.addFrame(fidx);
		}

		for (int tidx = 0; tidx < 20; tidx++) {
			BTrack bt = alg.scene.addTrack(0,0,1,1.0);
			// put each track into one frame
			alg.scene.addObservation(alg.scene.frames.get(tidx/2),bt,0,0);

			// with the exception of these tracks
			switch( tidx ) {
				case 3:
				case 8:
				case 18: alg.scene.addObservation(alg.scene.frames.get(6),bt,0,0); break;
			}
		}

		// let's make two tracks visual tracks
		alg.scene.tracks.get(10).visualTrack = new PointTrack();
		alg.scene.tracks.get(10).visualTrack.cookie = alg.scene.tracks.get(10);
		alg.scene.tracks.get(11).visualTrack = new PointTrack();
		alg.scene.tracks.get(11).visualTrack.cookie = alg.scene.tracks.get(11);

		// pick a few frames to drop
		GrowQueue_I32 drop = new GrowQueue_I32();
		drop.add(0);
		drop.add(4);
		drop.add(5);

		// run the function being tested and see if we get the expected results
		alg.dropFramesFromScene(drop);
		assertEquals(7,alg.scene.frames.size);
		// all tracks from frame 0 and 5 (4 total) should be removed. Only 1 from from 4
		assertEquals(15,alg.scene.tracks.size);
		// only two of the tracks were visual tracks that got removed
		assertEquals(2, alg.countDropVisual);
	}

	@Test
	void dropTracksNotVisibleAndTooFewObservations() {
		var alg = new BundleBase();
		alg.scene.addCamera(pinhole);
		alg.countDropVisual = 3; // we should discard a track if it has less than this observations

		// Create a set of frames that can view the tracks
		for (int fidx = 0; fidx < 5; fidx++) {
			alg.scene.addFrame(fidx);
		}

		// create 1/2 visible tracks with varying number of observations
		for (int tidx = 0; tidx < 10; tidx++) {
			BTrack bt = alg.scene.addTrack(0,0,tidx-1.5,1.0);
			bt.visualTrack = tidx%2==0 ? new PointTrack() : null;
			bt.id = tidx;
			int N = Math.min(tidx/2,alg.scene.frames.size);
			for (int fidx = 0; fidx < N; fidx++) {
				BFrame bf = alg.scene.frames.get(fidx);
				alg.scene.addObservation(bf,bt,0,0);
			}
		}

		alg.dropTracksNotVisibleAndTooFewObservations();
		alg.scene.sanityCheck();
		assertEquals(7,alg.scene.tracks.size);
		// all visible tracks should still be there
		int countVisible = 0;
		for (int i = 0; i < alg.scene.tracks.size; i++) {
			if( alg.scene.tracks.get(i).visualTrack != null )
				countVisible++;
		}
		assertEquals(5,countVisible);
		// Tracks which are not visible should be dropped
		for (int i = 0; i < alg.scene.tracks.size; i++) {
			BTrack bt = alg.scene.tracks.get(i);
			if( bt.visualTrack == null ) {
				assertTrue(bt.id>=6);
			}
		}
	}

	/**
	 * See if it drops tracks behind the camera
	 */
	@Test
	void dropBadBundleTracks_Behind() {
		var alg = new BundleBase();
		alg.scene.addCamera(pinhole);

		// Create frames at regular intervals along the z-axis
		for (int fidx = 0; fidx < 3; fidx++) {
			BFrame bf = alg.scene.addFrame(fidx);
			bf.frame_to_world.T.set(0,0,fidx);
		}

		// Create tracks at regular intervals along the z-axis and make each one visible by all frames
		for (int tidx = 0; tidx < 10; tidx++) {
			BTrack bt = alg.scene.addTrack(0,0,tidx-1.5,1.0);
			if( tidx%2 == 1 ) {
				// flip the sign to make sure that's handled correctly in the range check
				bt.worldLoc.scale(-1);
			}
			bt.id = tidx;
			for (int fidx = 0; fidx < alg.scene.frames.size ; fidx++) {
				BFrame bf = alg.scene.frames.get(fidx);
				alg.scene.addObservation(bf,bt,0,0);
			}
		}

		// Run the function being tested and examine the results
		alg.dropBadBundleTracks();
		alg.scene.sanityCheck();
		// A track must be in front of every camera to not be dropped
		assertEquals(6,alg.scene.tracks.size);
	}

	static class BundleBase extends VisOdomBundlePnPBase<BTrack> {
		public long frameID = 0;
		public int countDropVisual = 0;

		public BundleBase() {
			var bundleAdjustment = FactoryMultiView.bundleSparseMetric(null);
			scene = new VisOdomBundleAdjustment<>(bundleAdjustment,BTrack::new);
		}

		@Override
		protected void dropVisualTrack(PointTrack track) {
			countDropVisual++;
		}

		@Override
		public long getFrameID() {
			return frameID;
		}
	}
}