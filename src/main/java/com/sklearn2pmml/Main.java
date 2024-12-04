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
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.List;
import java.util.logging.LogManager;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.google.common.collect.LinkedHashMultiset;
import com.google.common.collect.Multiset;
import com.sun.istack.logging.Logger;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Version;
import org.jpmml.converter.Application;
import org.jpmml.converter.VersionConverter;
import org.jpmml.model.MarkupException;
import org.jpmml.model.PMMLOutputStream;
import org.jpmml.model.metro.MetroJAXBUtil;
import org.jpmml.model.visitors.VersionChecker;
import org.jpmml.model.visitors.VersionDowngrader;
import org.jpmml.model.visitors.VersionStandardizer;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.Storage;
import org.jpmml.python.StorageUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.EncodableUtil;
import org.jpmml.sklearn.SkLearnUtil;

public class Main extends Application {

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

	@Parameter (
		names = {"--pmml-schema", "--schema"},
		converter = VersionConverter.class
	)
	private Version version = null;


	static
	public void main(String... args) throws Exception {
		Main main = new Main();

		JCommander commander = JCommander.newBuilder()
			.addObject(main)
			.build();

		commander.parse(args);

		try {
			Application.setInstance(main);

			main.run();
		} finally {
			Application.setInstance(null);
		}
	}

	private void run() throws Exception {
		Object object;

		try(Storage storage = StorageUtil.createStorage(this.inputFile)){
			object = PickleUtil.unpickle(storage);
		}

		Encodable encodable = EncodableUtil.toEncodable(object);

		PMML pmml = encodable.encodePMML();

		if(!this.outputFile.exists()){
			File absoluteOutputFile = this.outputFile.getAbsoluteFile();

			File outputDir = absoluteOutputFile.getParentFile();
			if(!outputDir.exists()){
				outputDir.mkdirs();
			}
		} // End if

		if(this.version != null && this.version.compareTo(Version.XPMML) < 0){
			VersionStandardizer versionStandardizer = new VersionStandardizer();
			versionStandardizer.applyTo(pmml);

			VersionDowngrader versionDowngrader = new VersionDowngrader(this.version);
			versionDowngrader.applyTo(pmml);

			VersionChecker versionChecker = new VersionChecker(this.version);
			versionChecker.applyTo(pmml);

			List<MarkupException> exceptions = versionChecker.getExceptions();
			if(!exceptions.isEmpty()){
				Main.logger.severe("The PMML object has " + exceptions.size() + " incompatibilities with the requested PMML schema version:");

				Multiset<String> groupedMessages = LinkedHashMultiset.create();

				for(MarkupException exception : exceptions){
					groupedMessages.add(exception.getMessage());
				}

				Collection<Multiset.Entry<String>> entries = groupedMessages.entrySet();
				for(Multiset.Entry<String> entry : entries){
					Main.logger.warning(entry.getElement() + (entry.getCount() > 1 ? " (" + entry.getCount() + " cases)": ""));
				}
			}

			try(OutputStream os = new PMMLOutputStream(new FileOutputStream(this.outputFile), this.version)){
				MetroJAXBUtil.marshalPMML(pmml, os);
			}
		} else

		{
			try(OutputStream os = new FileOutputStream(this.outputFile)){
				MetroJAXBUtil.marshalPMML(pmml, os);
			}
		}
	}

	static {
		LogManager logManager = LogManager.getLogManager();

		try {
			logManager.readConfiguration(Main.class.getResourceAsStream("/logging.properties"));
		} catch(IOException ioe){
			ioe.printStackTrace(System.err);
		}
	}

	private static final Logger logger = Logger.getLogger(Main.class);

	static {
		SkLearnUtil.initOnce();
	}
}