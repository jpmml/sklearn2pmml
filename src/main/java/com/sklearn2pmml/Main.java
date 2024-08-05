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
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

import javax.xml.transform.TransformerFactory;
import javax.xml.transform.sax.SAXTransformerFactory;
import javax.xml.transform.sax.TransformerHandler;
import javax.xml.transform.stream.StreamResult;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Version;
import org.jpmml.converter.Application;
import org.jpmml.model.SAXUtil;
import org.jpmml.model.filters.ExportFilter;
import org.jpmml.model.metro.MetroJAXBUtil;
import org.jpmml.model.visitors.VersionInspector;
import org.jpmml.python.PickleUtil;
import org.jpmml.python.Storage;
import org.jpmml.python.StorageUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.EncodableUtil;
import org.jpmml.sklearn.SkLearnException;
import org.jpmml.sklearn.SkLearnUtil;
import org.xml.sax.InputSource;

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

		if(this.version != null && this.version.compareTo(Version.PMML_4_4) < 0){
			VersionInspector versionInspector = new VersionInspector();
			versionInspector.applyTo(pmml);

			Version minVersion = versionInspector.getMinimum();
			if(minVersion.compareTo(this.version) > 0){
				throw new SkLearnException("The generated markup requires PMML schema version " + minVersion.getVersion() + " or newer");
			}

			Version maxVersion = versionInspector.getMaximum();
			if(maxVersion.compareTo(this.version) < 0){
				throw new SkLearnException("The generated markup requires PMML schema version " + maxVersion.getVersion() + " or older");
			}
		} // End if

		if(!this.outputFile.exists()){
			File absoluteOutputFile = this.outputFile.getAbsoluteFile();

			File outputDir = absoluteOutputFile.getParentFile();
			if(!outputDir.exists()){
				outputDir.mkdirs();
			}
		}

		if(this.version != null && this.version.compareTo(Version.PMML_4_4) < 0){
			File tempFile = File.createTempFile("sklearn2pmml-", ".pmml");

			try(OutputStream os = new FileOutputStream(tempFile)){
				MetroJAXBUtil.marshalPMML(pmml, os);
			}

			SAXTransformerFactory transformerFactory = (SAXTransformerFactory)TransformerFactory.newInstance();

			try(OutputStream os = new FileOutputStream(this.outputFile)){
				TransformerHandler transformerHandler = transformerFactory.newTransformerHandler();
				transformerHandler.setResult(new StreamResult(os));

				ExportFilter exportFilter = new ExportFilter(SAXUtil.createXMLReader(), this.version);
				exportFilter.setContentHandler(transformerHandler);

				try(InputStream is = new FileInputStream(tempFile)){
					exportFilter.parse(new InputSource(is));
				}
			}

			tempFile.delete();
		} else

		{
			try(OutputStream os = new FileOutputStream(this.outputFile)){
				MetroJAXBUtil.marshalPMML(pmml, os);
			}
		}
	}

	static {
		SkLearnUtil.initOnce();
	}
}