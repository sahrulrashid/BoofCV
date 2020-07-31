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

package boofcv.alg.sfm.structure;

import boofcv.abst.geo.Estimate1ofTrifocalTensor;
import boofcv.abst.geo.bundle.BundleAdjustment;
import boofcv.abst.geo.bundle.PruneStructureFromSceneMetric;
import boofcv.abst.geo.bundle.SceneObservations;
import boofcv.abst.geo.bundle.SceneStructureMetric;
import boofcv.alg.geo.bundle.cameras.BundlePinholeSimplified;
import boofcv.alg.geo.selfcalib.*;
import boofcv.factory.geo.*;
import boofcv.misc.ConfigConverge;
import boofcv.struct.calib.CameraPinhole;
import boofcv.struct.geo.AssociatedTriple;
import georegression.struct.point.Point3D_F64;
import georegression.struct.se.Se3_F64;
import org.ddogleg.fitting.modelset.ransac.Ransac;
import org.ddogleg.optimization.lm.ConfigLevenbergMarquardt;
import org.ddogleg.struct.VerbosePrint;
import org.ejml.dense.row.CommonOps_DDRM;

import javax.annotation.Nullable;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static boofcv.alg.geo.MultiViewOps.triangulatePoints;

/**
 * Estimates the metric scene's structure given a set of sparse features associations from three views. This is
 * intended to give the best possible solution from the sparse set of matching features. Its internal
 * methods are updated as better strategies are found.
 *
 * Assumptions:
 * <ul>
 *     <li>Principle point is zero</li>
 *     <li>Zero skew</li>
 *     <li>fx = fy approximately</li>
 * </ul>
 *
 * The zero principle point is enforced prior to calling {@link #process} by subtracting the image center from
 * each pixel observations.
 *
 * Steps:
 * <ol>
 *     <li>Fit Trifocal tensor using RANSAC</li>
 *     <li>Get and refine camera matrices</li>
 *     <li>Compute dual absolute quadratic</li>
 *     <li>Estimate intrinsic parameters from DAC</li>
 *     <li>Estimate metric scene structure</li>
 *     <li>Sparse bundle adjustment</li>
 *     <li>Tweak parameters and sparse bundle adjustment again</li>
 * </ol>
 *
 * @author Peter Abeles
 */
public class ThreeViewEstimateMetricScene implements VerbosePrint {

	// Make all configurations public for ease of manipulation
	public ConfigRansac configRansac = new ConfigRansac();
	public ConfigTrifocal configTriRansac = new ConfigTrifocal();
	public ConfigTrifocal configTriFit = new ConfigTrifocal();
	public ConfigTrifocalError configError = new ConfigTrifocalError();
	public ConfigLevenbergMarquardt configLM = new ConfigLevenbergMarquardt();
	public ConfigBundleAdjustment configSBA = new ConfigBundleAdjustment();
	public ConfigConverge convergeSBA = new ConfigConverge(1e-6,1e-6,100);

	// estimating the trifocal tensor and storing which observations are in the inlier set
	public Ransac<MetricCameraTriple, AssociatedTriple> ransac;
	public List<AssociatedTriple> inliers;

	// how much and where it should print to
	private PrintStream verbose;

	// storage for pinhole cameras
	protected List<CameraPinhole> listPinhole = new ArrayList<>();

	// Refines the structure
	public BundleAdjustment<SceneStructureMetric> bundleAdjustment;

	// Bundle adjustment data structure and tuning parameters
	public SceneStructureMetric structure;
	public SceneObservations observations;

	// If a positive number the focal length will be assumed to be that
	public double manualFocalLength=-1;

	// How many features it will keep when pruning
	public double pruneFraction = 0.7;

	// shape of input images.
	private int width, height; // TODO Get size for each image individually

	// metric location of each camera. The first view is always identity
	protected List<Se3_F64> worldToView = new ArrayList<>();

	/**
	 * Sets configurations to their default value
	 */
	public ThreeViewEstimateMetricScene() {
		configRansac.iterations = 500;
		configRansac.inlierThreshold = 1;

		configError.model = ConfigTrifocalError.Model.REPROJECTION_REFINE;

		configTriFit.which = EnumTrifocal.ALGEBRAIC_7;
		configTriFit.converge.maxIterations = 100;

		configLM.dampeningInitial = 1e-3;
		configLM.hessianScaling = false;
		configSBA.configOptimizer = configLM;

		for (int i = 0; i < 3; i++) {
			worldToView.add( new Se3_F64());
		}
	}

	/**
	 * Determines the metric scene. The principle point is assumed to be zero in the passed in pixel coordinates.
	 * Typically this is done by subtracting the image center from each pixel coordinate for each view.
	 *
	 * @param associated List of associated features from 3 views. pixels
	 * @param width width of all images
	 * @param height height of all images
	 * @return true if successful or false if it failed
	 */
	public boolean process(List<AssociatedTriple> associated , int width , int height ) {
		init(width, height);

		// Fit a trifocal tensor to the input observations
		if (!robustMetric(associated) )
			return false;

		// Run bundle adjustment while make sure a valid solution is found
		setupMetricBundleAdjustment(inliers);

		bundleAdjustment = FactoryMultiView.bundleSparseMetric(configSBA);
		findBestValidSolution(bundleAdjustment);

		// Prune outliers and run bundle adjustment one last time
		pruneOutliers(bundleAdjustment);

		return true;
	}

	private void init( int width , int height) {
		this.width = width;
		this.height = height;
		Estimate1ofTrifocalTensor trifocal = FactoryMultiView.trifocal_1(null);
		var alg = new SelfCalibrationLinearDualQuadratic(1.0);

		int maxSide = Math.max(width,height);
		var brute = new TrifocalBruteForceSelfCalibration();
		brute.configure(maxSide/3,maxSide*2);
		brute.numberOfSamples = 50;
		brute.fixedFocus = true;

//		var generator = new GenerateMetricCameraTripleDualQuadratic(trifocal,alg);
		var generator = new GenerateMetricCameraTripleBruteForceFocus(trifocal,brute);
		var distance = new DistanceMetricTripleReprojection23();
		var manager = new ModelManagerMetricCameraTriple();

		// convert from pixels to pixels squared
		double threshold = 2*Math.pow(2.5,2);

		ransac = new Ransac<>(0xDeADBEED, manager, generator, distance, 7500, threshold);
		structure = null;
		observations = null;
	}

	/**
	 * Fits a trifocal tensor to the list of matches features using a robust method
	 */
	private boolean robustMetric(List<AssociatedTriple> associated) {
		// Fit a trifocal tensor to the observations robustly
		if( !ransac.process(associated) )
			return false;

		inliers = ransac.getMatchSet();
		MetricCameraTriple model = ransac.getModelParameters();
		if( verbose != null )
			verbose.println("Remaining after RANSAC "+inliers.size()+" / "+associated.size());

		worldToView.get(0).reset();
		worldToView.get(1).set(model.view_1_to_2);
		worldToView.get(2).set(model.view_1_to_3);

		listPinhole.clear();
		listPinhole.add(model.view1);
		listPinhole.add(model.view2);
		listPinhole.add(model.view3);

		// scale is arbitrary. Set max translation to 1
		double maxT = 0;
		for( Se3_F64 p : worldToView ) {
			maxT = Math.max(maxT,p.T.norm());
		}
		for( Se3_F64 p : worldToView ) {
			p.T.scale(1.0/maxT);
			if( verbose != null ) {
				verbose.println(p);
			}
		}

		return true;
	}

	/**
	 * Prunes the features with the largest reprojection error
	 */
	private void pruneOutliers(BundleAdjustment<SceneStructureMetric> bundleAdjustment) {
		// see if it's configured to not prune
		if( pruneFraction == 1.0 )
			return;
		PruneStructureFromSceneMetric pruner = new PruneStructureFromSceneMetric(structure,observations);
		pruner.pruneObservationsByErrorRank(pruneFraction);
		pruner.pruneViews(10);
		pruner.prunePoints(1);
		bundleAdjustment.setParameters(structure,observations);
		bundleAdjustment.optimize(structure);

		if( verbose != null ) {
			verbose.println("\nCamera");
			for (int i = 0; i < structure.cameras.size; i++) {
				verbose.println(structure.cameras.data[i].getModel().toString());
			}
			verbose.println("\n\nworldToView");
			for (int i = 0; i < structure.views.size; i++) {
				verbose.println(structure.views.data[i].worldToView.toString());
			}
			verbose.println("Fit Score: " + bundleAdjustment.getFitScore());
		}
	}

	/**
	 * Tries a bunch of stuff to ensure that it can find the best solution which is physically possible
	 */
	private void findBestValidSolution(BundleAdjustment<SceneStructureMetric> bundleAdjustment) {
		// prints out useful debugging information that lets you know how well it's converging
		bundleAdjustment.setVerbose(verbose,null);

		// Specifies convergence criteria
		bundleAdjustment.configure(convergeSBA.ftol, convergeSBA.gtol, convergeSBA.maxIterations);

		bundleAdjustment.setParameters(structure,observations);
		bundleAdjustment.optimize(structure);

		// ensure that the points are in front of the camera and are a valid solution
		if( checkBehindCamera(structure) ) {
			if( verbose != null )
				verbose.println("  flipping view");
			flipAround(structure,observations);
			bundleAdjustment.setParameters(structure,observations);
			bundleAdjustment.optimize(structure);
		}

		double bestScore = bundleAdjustment.getFitScore();
		List<Se3_F64> bestPose = new ArrayList<>();
		List<BundlePinholeSimplified> bestCameras = new ArrayList<>();
		for (int i = 0; i < structure.views.size; i++) {
			BundlePinholeSimplified c = structure.cameras.data[i].getModel();
			bestPose.add(structure.views.data[i].worldToView.copy());
			bestCameras.add( c.copy());
		}

		for (int i = 0; i < structure.cameras.size; i++) {
			BundlePinholeSimplified c = structure.cameras.data[i].getModel();
			c.f = listPinhole.get(i).fx;
			c.k1 = c.k2 = 0;
		}
		// flip rotation assuming that it was done wrong
		for (int i = 1; i < structure.views.size; i++) {
			CommonOps_DDRM.transpose(structure.views.data[i].worldToView.R);
		}
		triangulatePoints(structure,observations);

		bundleAdjustment.setParameters(structure,observations);
		bundleAdjustment.optimize(structure);

		if( checkBehindCamera(structure) ) {
			if( verbose != null )
				verbose.println("  flipping view");
			flipAround(structure,observations);
			bundleAdjustment.setParameters(structure,observations);
			bundleAdjustment.optimize(structure);
		}

		// revert to old settings
		if( verbose != null )
			verbose.println(" ORIGINAL / NEW = " + bestScore+" / "+bundleAdjustment.getFitScore());
		if( bundleAdjustment.getFitScore() > bestScore ) {
			if( verbose != null )
				verbose.println("  recomputing old structure");
			for (int i = 0; i < structure.cameras.size; i++) {
				BundlePinholeSimplified c = structure.cameras.data[i].getModel();
				c.set(bestCameras.get(i));
				structure.views.data[i].worldToView.set(bestPose.get(i));
			}
			triangulatePoints(structure,observations);
			bundleAdjustment.setParameters(structure,observations);
			bundleAdjustment.optimize(structure);
			if( verbose != null )
				verbose.println("  score = "+bundleAdjustment.getFitScore());
		}
	}

	/**
	 * Using the initial metric reconstruction, provide the initial configurations for bundle adjustment
	 */
	private void setupMetricBundleAdjustment(List<AssociatedTriple> inliers) {
		// Construct bundle adjustment data structure
		structure = new SceneStructureMetric(false);
		structure.initialize(3,3,inliers.size());
		observations = new SceneObservations();
		observations.initialize(3);

		for (int i = 0; i < listPinhole.size(); i++) {
			CameraPinhole cp = listPinhole.get(i);
			BundlePinholeSimplified bp = new BundlePinholeSimplified();

			bp.f = cp.fx;

			structure.setCamera(i,false,bp);
			structure.setView(i,i==0,worldToView.get(i));
			structure.connectViewToCamera(i,i);
		}
		for (int i = 0; i < inliers.size(); i++) {
			AssociatedTriple t = inliers.get(i);

			observations.getView(0).add(i,(float)t.p1.x,(float)t.p1.y);
			observations.getView(1).add(i,(float)t.p2.x,(float)t.p2.y);
			observations.getView(2).add(i,(float)t.p3.x,(float)t.p3.y);

			structure.connectPointToView(i,0);
			structure.connectPointToView(i,1);
			structure.connectPointToView(i,2);
		}
		// Initial estimate for point 3D locations
		triangulatePoints(structure,observations);
	}

	/**
	 * Checks to see if a solution was converged to where the points are behind the camera. This is
	 * pysically impossible
	 */
	private boolean checkBehindCamera(SceneStructureMetric structure ) {

		int totalBehind = 0;
		Point3D_F64 X = new Point3D_F64();
		for (int i = 0; i < structure.points.size; i++) {
			structure.points.data[i].get(X);
			if( X.z < 0 )
				totalBehind++;
		}

		if( verbose != null ) {
			verbose.println("points behind "+totalBehind+" / "+structure.points.size);
		}

		return totalBehind > structure.points.size/2;
	}

	/**
	 * Flip the camera pose around. This seems to help it converge to a valid solution if it got it backwards
	 * even if it's not technically something which can be inverted this way
	 */
	private static void flipAround(SceneStructureMetric structure, SceneObservations observations) {
		// The first view will be identity
		for (int i = 1; i < structure.views.size; i++) {
			Se3_F64 w2v = structure.views.data[i].worldToView;
			w2v.set(w2v.invert(null));
		}
		triangulatePoints(structure,observations);
	}

	public SceneStructureMetric getStructure() {
		return structure;
	}

	@Override
	public void setVerbose(@Nullable PrintStream out, @Nullable Set<String> configuration) {
		this.verbose = out;
	}
}
