/*
 * Copyright (c) 2011-2019, Peter Abeles. All Rights Reserved.
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

package boofcv.alg.fiducial.calib.chess;

import boofcv.alg.feature.detect.chess.ChessboardCorner;
import boofcv.alg.interpolate.InterpolatePixelS;
import boofcv.factory.interpolate.FactoryInterpolation;
import boofcv.misc.BoofMiscOps;
import boofcv.struct.border.BorderType;
import boofcv.struct.image.ImageGray;
import georegression.metric.UtilAngle;
import org.ddogleg.sorting.QuickSelect;

/**
 * Computes edge intensity for the line between two corners. Assumes the edge is approximately straight. This means
 * lens distortion between two points can't be too massive. Orientation of corners is used to estimate which
 * side of the line should be white/black
 *
 * @author Peter Abeles
 */
public class ChessboardCornerEdgeIntensity<T extends ImageGray<T>> {

	// used to sample image at non integer points and handles boundary conditions
	InterpolatePixelS<T> interpolate;

	/**
	 * Number of points along the line from corner a to b that will be sampled
	 */
	private int lengthSamples=10;
	private float[] sampleValues = new float[lengthSamples];

	/**
	 * Number of points radially outwards along the line that are sampled
	 */
	int tangentSamples=2;

	/**
	 * Number of samples along each line when computing x-corner score
	 */
	int xcornerSamples=4;

	// find the normal pointing towards white. Magnitude is relative to distance between two corners
	float nx,ny;
	// tangent step
	float tx,ty;
	float normalDiv =20.0f;

	// length of the line segment between the two points
	float lineLength;

	int width,height;

	public ChessboardCornerEdgeIntensity( Class<T> imageType ) {
		interpolate = FactoryInterpolation.bilinearPixelS(imageType, BorderType.EXTENDED);
	}

	public void setImage( T image ) {
		interpolate.setImage(image);
		this.width = image.width;
		this.height = image.height;
	}

	/**
	 * Computes a value for the line intensity between two corners.
	 *
	 * @param ca corner a
	 * @param cb corner b
	 * @param direction_a_to_b Direction from a to b in radians.
	 * @return the line intensity. more positive more intense. Units = pixels intensity
	 */
	public float process( ChessboardCorner ca , ChessboardCorner cb , double direction_a_to_b ) {
//		if( ca.distance(510,395) < 1.5 && cb.distance(541,375) < 2)
//			System.out.println("stop for intensity");

		float cx = (float)ca.x;
		float cy = (float)ca.y;
		float dx = (float)(cb.x-ca.x);
		float dy = (float)(cb.y-ca.y);

		// find the direction that it should be and the magnitude of the step in tangential direction
		computeUnitNormal(ca, direction_a_to_b, dx, dy);

		// step away from the corner points. This is only really important with small chessboard where the samples
		// will put it next to the corner
		if( lineLength > 2f ) {
			cx += nx;
			cy += ny;
			dx -= 2 * nx;
			dy -= 2 * ny;
		}

//		if( ca.distance(1041,80) < 2 && cb.distance(1294,232) < 2 )
//			System.out.println("Egads");

		// move from one side to the other
		// divide it into lengthSamples+1 regions and don't sample the tail ends
		for (int i = 1; i <= lengthSamples; i++) {
			float f = ((float)i)/(lengthSamples+1);
			float x0 = cx+dx*f;
			float y0 = cy+dy*f;

			float maxValue = -Float.MAX_VALUE;

			for (int l = 1; l <= tangentSamples; l++) {
				float white = interpolate.get(x0-ty*l,y0+tx*l);
				float black = interpolate.get(x0+ty*l,y0-tx*l);

				maxValue = Math.max(maxValue,white-black);
			}

			sampleValues[i-1] = maxValue;
		}

		// Select one of the most intense values.
		// Originally min was used but that proved too sensitive to outliers
		return QuickSelect.select(sampleValues,2,lengthSamples);
	}

	/**
	 * Compute the X-Corner score with sampling distance scaled by the length of the line
	 */
	public boolean computeXCorner( ChessboardCorner corner ) {

		float c = (float)Math.cos(corner.orientation);
		float s = (float)Math.sin(corner.orientation);

		// Two scenarios need to be considered when selecting the length of the x-corner.
		// 1) Inside the chessboard.
		// 2) Corner at the border of the chessboard.
		//
		// When inside it can be 1/2 the length when you take in account strong perspective and lens distortion.
		// However the outside border is different because some people create chessboards with very small outside
		// squares/rectangles
		float l = (float)(Math.max(6.0,lineLength/3)/xcornerSamples);
		c *= l;
		s *= l;

		float xx = (float)corner.x;
		float yy = (float)corner.y;

//		if( corner.distance(893,12) < 2 && lineLength < 250)
//			System.out.println("XCorner start");

//		System.out.println("ADSASDASDSAD------------------------");
		float white0 = lineSampleForXCorner(s,-c,xx,yy);
		float white1 = lineSampleForXCorner(-s,c,xx,yy);
		float black0 = lineSampleForXCorner(c,s,xx,yy);
		float black1 = lineSampleForXCorner(-c,-s,xx,yy);

		float mean = (white0+white1+black0+black1)/4.0f;

		white0 -= mean;
		white1 -= mean;
		black0 -= mean;
		black1 -= mean;

//		if( corner.distance(893,12) < 2 )
//			System.out.println("XCorner length = "+lineLength+" colors = "+white0+" "+white1+" "+black0+" "+black1);

		return white0 > 0 && white1 > 0 && black0 < 0 && black1 < 0;
	}

	private float lineSampleForXCorner(float c , float s , float x0 , float y0  ) {
		float sum = 0;
		int count = 0;
		// skip the point closest to the corner. intensity is ambiguous there
		for (int l = 1; l <= xcornerSamples; l++) {
			float x = x0 + c*l;
			float y = y0 + s*l;
			if(BoofMiscOps.checkInside(width,height,x,y) ) {
//			System.out.println("value["+l+"] = "+interpolate.get(x0+c*l,y0+s*l));
				sum += interpolate.get(x0 + c * l, y0 + s * l);
				count++;
			}
		}

		if( count == 0)
			return 0;
		else
			return sum / count;
	}

	/**
	 * Finds the line's unit normal and make sure it points towards what should be white pixels
	 * @param ca corner a
	 * @param direction_a_to_b direction from corner a to b. radians. -pi to pi
	 * @param dx b.x - a.x
	 * @param dy b.y - a.y
	 */
	void computeUnitNormal(ChessboardCorner ca, double direction_a_to_b, float dx, float dy) {
		lineLength = (float)Math.sqrt(dx*dx + dy*dy);

		// it will now have a normal of 1
		nx = tx = dx/lineLength;
		ny = ty = dy/lineLength;

		// set the magnitude relative to the square size. Blurred images won't have sharp edges
		// at the same time the magnitude of |n| shouldn't be less than 1
		float sampleLength = Math.max(1f,lineLength/normalDiv);
		tx *= sampleLength;
		ty *= sampleLength;

		double dir0 = UtilAngle.boundHalf(direction_a_to_b);
		double dir1 = UtilAngle.boundHalf(ca.orientation-Math.PI/4);
		if(UtilAngle.distHalf(dir0,dir1) < Math.PI/4.0 ) {
			tx = -tx;
			ty = -ty;
		}
	}

	public int getLengthSamples() {
		return lengthSamples;
	}

	public void setLengthSamples(int lengthSamples) {
		this.lengthSamples = lengthSamples;
		if( sampleValues.length < lengthSamples)
			sampleValues = new float[lengthSamples];
	}

	public int getTangentSamples() {
		return tangentSamples;
	}

	public void setTangentSamples(int tangentSamples) {
		this.tangentSamples = tangentSamples;
	}

	public Class<T> getImageType() {
		return interpolate.getImageType().getImageClass();
	}
}
