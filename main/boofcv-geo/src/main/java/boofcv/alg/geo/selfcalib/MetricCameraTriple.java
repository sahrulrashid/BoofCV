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
import georegression.struct.se.Se3_F64;

/**
 * Results of upgrading a three view scenario from a projective into a metric scene. The first camera view
 * is the origin of this coordinate system and a {@link CameraPinhole} model is used.
 *
 * @author Peter Abeles
 */
public class MetricCameraTriple {
	// rigid body transforms from each view. View 1 is assumed to be the origin and is identity
	public final Se3_F64 view_1_to_2 = new Se3_F64();
	public final Se3_F64 view_1_to_3 = new Se3_F64();

	// Intrinsic camera model for each view
	public final CameraPinhole view1 = new CameraPinhole();
	public final CameraPinhole view2 = new CameraPinhole();
	public final CameraPinhole view3 = new CameraPinhole();
}
