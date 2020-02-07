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

package boofcv.io.video;

import boofcv.io.UtilIO;
import boofcv.io.image.SimpleImageSequence;
import boofcv.io.wrapper.images.ImageStreamSequence;
import boofcv.io.wrapper.images.JpegByteImageSequence;
import boofcv.struct.image.ImageBase;
import boofcv.struct.image.ImageType;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.net.URL;
import java.util.List;

/**
 * This video interface attempts to load a native reader.  If that fails, jcodec, if that fails it just
 * uses the built in video types.
 *
 * @author Peter Abeles
 */
public class DynamicVideoInterface implements VideoInterface {

	VideoInterface jcodec;
	VideoInterface ffmpeg;
	BoofMjpegVideo mjpeg = new BoofMjpegVideo();

	public DynamicVideoInterface() {
		try {
			ffmpeg = loadManager("boofcv.io.ffmpeg.FfmpegVideo");
		} catch( RuntimeException ignore ) {}

		try {
			jcodec = loadManager("boofcv.io.jcodec.JCodecVideoInterface");
		} catch( RuntimeException ignore ) {}
	}

	@Override
	public <T extends ImageBase<T>> SimpleImageSequence<T> load(String fileName, ImageType<T> imageType) {
		URL url = UtilIO.ensureURL(fileName);
		if( url == null )
			throw new RuntimeException("Can't open "+fileName);

		InputStream stream=null;
		try {
			stream = url.openStream();

			// Use built in movie readers for these file types
			if( fileName.endsWith("mjpeg") || fileName.endsWith("MJPEG") ||
					fileName.endsWith("mjpg") || fileName.endsWith("MJPG") ) {
				VideoMjpegCodec codec = new VideoMjpegCodec();
				List<byte[]> data = codec.read(stream);
				return new JpegByteImageSequence<>(imageType, data, false);
			} else if( fileName.endsWith("mpng") || fileName.endsWith("MPNG")) {
				return new ImageStreamSequence<>(stream, true, imageType);
			}

			try {
				if( ffmpeg != null ) {
					return ffmpeg.load(fileName, imageType);
				}
			} catch( RuntimeException e ){
//				e.printStackTrace();
				return null;
			}

			try {
				if( jcodec != null ) {
					if( fileName.endsWith(".mp4") || fileName.endsWith(".MP4")) {
						return jcodec.load(fileName,imageType);
					}
				}
			} catch( RuntimeException ignore ){}
			System.err.println("No working codec found for file: " + fileName);

		} catch (IOException e) {
			System.err.println("Error opening. "+e.getMessage());
			return null;
		} finally {
			if( stream != null ) {
				try {
					stream.close();
				} catch (IOException e) {
					System.err.println("Failed to close stream. "+e.getMessage());
				}
			}
		}

		return null;
	}

	/**
	 * Loads the specified default {@link VideoInterface}.
	 *
	 * @return Video interface
	 */
	public static VideoInterface loadManager( String pathToManager ) {
		try {
			Class c = Class.forName(pathToManager);
			return (VideoInterface) c.getConstructor().newInstance();
		} catch (ClassNotFoundException e) {
			throw new RuntimeException("Class not found.  Is it included in the class path?");
		} catch (InstantiationException | IllegalAccessException | NoSuchMethodException | InvocationTargetException e) {
			throw new RuntimeException(e);
		}
	}
}
