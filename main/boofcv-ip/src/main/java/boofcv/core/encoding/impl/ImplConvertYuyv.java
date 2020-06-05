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

package boofcv.core.encoding.impl;

import boofcv.struct.image.*;

//CONCURRENT_INLINE import boofcv.concurrency.BoofConcurrency;

/**
 *
 * Implementation of {@link boofcv.core.encoding.ConvertYuyv}
 *
 * @author Peter Abeles
 */
public class ImplConvertYuyv {

	public static void yuyvToPlanarRgb_U8(byte[] dataYV, Planar<GrayU8> output) {

		GrayU8 R = output.getBand(0);
		GrayU8 G = output.getBand(1);
		GrayU8 B = output.getBand(2);

		final int yStride = output.width*2;

		//CONCURRENT_BELOW BoofConcurrency.loopFor(0, output.height, row -> {
		for( int row = 0; row < output.height; row++ ) {
			int indexY = row*yStride;
			int indexU = indexY+1;
			int indexOut = output.startIndex + row*output.stride;

			for( int col = 0; col < output.width; col++, indexY += 2, indexOut++ ) {
				int y = 1191*((dataYV[indexY] & 0xFF) - 16);
				int cb = (dataYV[ indexU ] & 0xFF) - 128;
				int cr = (dataYV[ indexU+2] & 0xFF) - 128;

				y = ((y >>> 31)^1)*y;

				int r = (y + 1836*cr) >> 10;
				int g = (y - 547*cr - 218*cb) >> 10;
				int b = (y + 2165*cb) >> 10;

				r *= ((r >>> 31)^1);
				g *= ((g >>> 31)^1);
				b *= ((b >>> 31)^1);

				if( r > 255 ) r = 255;
				if( g > 255 ) g = 255;
				if( b > 255 ) b = 255;

				R.data[indexOut] = (byte)r;
				G.data[indexOut] = (byte)g;
				B.data[indexOut] = (byte)b;

				indexU += 4*(col&0x1);
			}
		}
		//CONCURRENT_ABOVE });
	}

	public static void yuyvToPlanarRgb_F32(byte[] dataYV, Planar<GrayF32> output) {

		GrayF32 R = output.getBand(0);
		GrayF32 G = output.getBand(1);
		GrayF32 B = output.getBand(2);

		final int yStride = output.width*2;

		//CONCURRENT_BELOW BoofConcurrency.loopFor(0, output.height, row -> {
		for( int row = 0; row < output.height; row++ ) {
			int indexY = row*yStride;
			int indexU = indexY+1;
			int indexOut = output.startIndex + row*output.stride;

			for( int col = 0; col < output.width; col++, indexY += 2, indexOut++ ) {
				int y = 1191*((dataYV[indexY] & 0xFF) - 16);
				int cb = (dataYV[ indexU ] & 0xFF) - 128;
				int cr = (dataYV[ indexU+2] & 0xFF) - 128;

				y = ((y >>> 31)^1)*y;

				int r = (y + 1836*cr) >> 10;
				int g = (y - 547*cr - 218*cb) >> 10;
				int b = (y + 2165*cb) >> 10;

				r *= ((r >>> 31)^1);
				g *= ((g >>> 31)^1);
				b *= ((b >>> 31)^1);

				if( r > 255 ) r = 255;
				if( g > 255 ) g = 255;
				if( b > 255 ) b = 255;

				R.data[indexOut] = r;
				G.data[indexOut] = g;
				B.data[indexOut] = b;

				indexU += 4*(col&0x1);
			}
		}
		//CONCURRENT_ABOVE });
	}

	public static void yuyvToInterleaved(byte[] dataYV, InterleavedU8 output) {
		final int yStride = output.width*2;

		//CONCURRENT_BELOW BoofConcurrency.loopFor(0, output.height, row -> {
		for( int row = 0; row < output.height; row++ ) {
			int indexY = row*yStride;
			int indexU = indexY+1;
			int indexOut = output.startIndex + row*output.stride;

			for( int col = 0; col < output.width; col++, indexY += 2 ) {
				int y = 1191*((dataYV[indexY] & 0xFF) - 16);
				int cb = (dataYV[ indexU ] & 0xFF) - 128;
				int cr = (dataYV[ indexU+2] & 0xFF) - 128;

				y = ((y >>> 31)^1)*y;

				int r = (y + 1836*cr) >> 10;
				int g = (y - 547*cr - 218*cb) >> 10;
				int b = (y + 2165*cb) >> 10;

				r *= ((r >>> 31)^1);
				g *= ((g >>> 31)^1);
				b *= ((b >>> 31)^1);


				if( r > 255 ) r = 255;
				if( g > 255 ) g = 255;
				if( b > 255 ) b = 255;

				output.data[indexOut++] = (byte)r;
				output.data[indexOut++] = (byte)g;
				output.data[indexOut++] = (byte)b;

				indexU += 4*(col&0x1);
			}
		}
		//CONCURRENT_ABOVE });
	}

	public static void yuyvToInterleaved(byte[] dataYV, InterleavedF32 output) {
		final int yStride = output.width*2;

		//CONCURRENT_BELOW BoofConcurrency.loopFor(0, output.height, row -> {
		for( int row = 0; row < output.height; row++ ) {
			int indexY = row*yStride;
			int indexU = indexY+1;
			int indexOut = output.startIndex + row*output.stride;

			for( int col = 0; col < output.width; col++, indexY += 2 ) {
				int y = 1191*((dataYV[indexY] & 0xFF) - 16);
				int cb = (dataYV[ indexU ] & 0xFF) - 128;
				int cr = (dataYV[ indexU+2] & 0xFF) - 128;

				y = ((y >>> 31)^1)*y;

				int r = (y + 1836*cr) >> 10;
				int g = (y - 547*cr - 218*cb) >> 10;
				int b = (y + 2165*cb) >> 10;

				r *= ((r >>> 31)^1);
				g *= ((g >>> 31)^1);
				b *= ((b >>> 31)^1);


				if( r > 255 ) r = 255;
				if( g > 255 ) g = 255;
				if( b > 255 ) b = 255;

				output.data[indexOut++] = r;
				output.data[indexOut++] = g;
				output.data[indexOut++] = b;

				indexU += 4*(col&0x1);
			}
		}
		//CONCURRENT_ABOVE });
	}

	public static void yuyvToGray(byte[] dataYV, GrayU8 output) {
		final int yStride = output.width*2;

		//CONCURRENT_BELOW BoofConcurrency.loopFor(0, output.height, row -> {
		for( int row = 0; row < output.height; row++ ) {
			int indexY = row*yStride;
			int indexOut = output.startIndex + row*output.stride;

			for( int col = 0; col < output.width; col++, indexY += 2 ) {
				output.data[indexOut++] = dataYV[indexY];
			}
		}
		//CONCURRENT_ABOVE });
	}

	public static void yuyvToGray(byte[] dataYV, GrayF32 output) {
		final int yStride = output.width*2;

		//CONCURRENT_BELOW BoofConcurrency.loopFor(0, output.height, row -> {
		for( int row = 0; row < output.height; row++ ) {
			int indexY = row*yStride;
			int indexOut = output.startIndex + row*output.stride;

			for( int col = 0; col < output.width; col++, indexY += 2 ) {
				output.data[indexOut++] = dataYV[indexY]&0xFF;
			}
		}
		//CONCURRENT_ABOVE });
	}
}