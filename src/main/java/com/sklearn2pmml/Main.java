/*
 * Copyright (c) 2022 Villu Ruusmann
 *
 * This file is part of SkLearn2PMML
 *
 * SkLearn2PMML is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SkLearn2PMML is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with SkLearn2PMML.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.sklearn2pmml;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.dmg.pmml.PMML;
import org.jpmml.model.metro.MetroJAXBUtil;
import org.jpmml.python.CastFunction;
import org.jpmml.python.ClassDictUtil;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.Storage;
import org.jpmml.python.StorageUtil;
import org.jpmml.sklearn.SkLearnEncoder;
import sklearn2pmml.pipeline.PMMLPipeline;

public class Main {

	@Parameter (
		names = {"--pkl-pipeline-input", "--pkl-input"},
		required = true
	)
	private File inputFile = null;

	@Parameter (
		names = {"--pmml-output"},
		required = true
	)
	private File outputFile = null;


	static
	public void main(String... args) throws Exception {
		Main main = new Main();

		JCommander commander = JCommander.newBuilder()
			.addObject(main)
			.build();

		commander.parse(args);

		main.run();
	}

	private void run() throws Exception {
		SkLearnEncoder encoder = new SkLearnEncoder();

		Object object;

		try(Storage storage = StorageUtil.createStorage(this.inputFile)){
			object = PickleUtil.unpickle(storage);
		}

		CastFunction<PMMLPipeline> castFunction = new CastFunction<PMMLPipeline>(PMMLPipeline.class){

			@Override
			protected String formatMessage(Object object){
				return "The unpickled object (" + ClassDictUtil.formatClass(object) + ") is not a PMMLPipeline";
			}
		};

		PMMLPipeline pmmlPipeline = castFunction.apply(object);

		PMML pmml = pmmlPipeline.encodePMML(encoder);

		if(!this.outputFile.exists()){
			File absoluteOutputFile = this.outputFile.getAbsoluteFile();

			File outputDir = absoluteOutputFile.getParentFile();
			if(!outputDir.exists()){
				outputDir.mkdirs();
			}
		}

		try(OutputStream os = new FileOutputStream(this.outputFile)){
			MetroJAXBUtil.marshalPMML(pmml, os);
		}
	}
}