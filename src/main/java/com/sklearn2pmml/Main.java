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
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.jar.Attributes;
import java.util.jar.Manifest;
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
import org.jpmml.model.JAXBSerializer;
import org.jpmml.model.MarkupException;
import org.jpmml.model.PMMLOutputStream;
import org.jpmml.model.metro.MetroJAXBSerializer;
import org.jpmml.model.visitors.VersionChecker;
import org.jpmml.model.visitors.VersionDowngrader;
import org.jpmml.model.visitors.VersionStandardizer;
import org.jpmml.python.PythonUnpickler;
import org.jpmml.python.Storage;
import org.jpmml.python.StorageUtil;
import org.jpmml.sklearn.Encodable;
import org.jpmml.sklearn.EncodableUtil;
import org.jpmml.sklearn.SkLearnUnpickler;
import org.jpmml.sklearn.SkLearnUtil;
import org.jpmml.telemetry.Incident;
import org.jpmml.telemetry.TelemetryClient;

public class Main extends Application {

	@Parameter (
		names = {"--pkl-pipeline-input", "--pkl-input"},
		required = true,
		order = 1
	)
	private File inputFile = null;

	@Parameter (
		names = {"--pmml-output"},
		required = true,
		order = 2
	)
	private File outputFile = null;

	@Parameter (
		names = {"--pmml-schema", "--schema"},
		converter = VersionConverter.class,
		order = 3
	)
	private Version version = null;


	static
	public void main(String... args) throws Exception {
		Main main = new Main();

		JCommander commander = JCommander.newBuilder()
			.addObject(main)
			.build();

		commander.parse(args);

		main.validate();

		try {
			Application.setInstance(main);

			main.run();
		} catch(FileNotFoundException fnfe){
			throw fnfe;
		} catch(Exception e){
			Package _package = Main.class.getPackage();

			Map<String, String> environment;

			String sklearn2pmmlEnvironment = System.getenv("SKLEARN2PMML_ENVIRONMENT");
			if(sklearn2pmmlEnvironment != null && sklearn2pmmlEnvironment.length() > 0){
				environment = parseEnvironment(sklearn2pmmlEnvironment);
			} else

			{
				environment = new LinkedHashMap<>();
				environment.put(System.getProperty("java.vendor"), System.getProperty("java.version"));
				environment.put("sklearn2pmml", _package.getImplementationVersion());
			}

			Incident incident = new Incident()
				.setProject("sklearn2pmml")
				.setEnvironment(environment)
				.setException(e);

			try {
				TelemetryClient.report("https://telemetry.jpmml.org/v1/incidents", incident);
			} catch(IOException ioe){
				// Ignored
			}

			throw e;
		} finally {
			Application.setInstance(null);
		}
	}

	private void validate(){
		Instant buildTimestamp = getBuildTimestamp();

		if(buildTimestamp != null){
			Instant now = Instant.now();

			Instant updateRequiredTimestamp = now.minus(12 * 30, ChronoUnit.DAYS);
			if(buildTimestamp.isBefore(updateRequiredTimestamp)){
				logger.severe("The SkLearn2PMML package is older than 12 months and must be updated");

				throw new RuntimeException("The SkLearn2PMML package has expired");
			}

			Instant updateRecommendedTimestamp = now.minus(6 * 30, ChronoUnit.DAYS);
			if(buildTimestamp.isBefore(updateRecommendedTimestamp)){
				logger.warning("The SkLearn2PMML package is older than 6 months and should be updated");
			}
		}
	}

	private void run() throws Exception {
		Object object;

		try(Storage storage = StorageUtil.createStorage(this.inputFile)){
			PythonUnpickler pythonUnpickler = new SkLearnUnpickler();

			object = pythonUnpickler.load(storage);
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

			JAXBSerializer jaxbSerializer = new MetroJAXBSerializer();

			try(OutputStream os = new PMMLOutputStream(new FileOutputStream(this.outputFile), this.version)){
				jaxbSerializer.serializePretty(pmml, os);
			}
		} else

		{
			JAXBSerializer jaxbSerializer = new MetroJAXBSerializer();

			try(OutputStream os = new FileOutputStream(this.outputFile)){
				jaxbSerializer.serializePretty(pmml, os);
			}
		}
	}

	private Instant getBuildTimestamp(){
		Manifest manifest = getManifest();

		Attributes mainAttributes = manifest.getMainAttributes();

		String buildTimestampString = mainAttributes.getValue("Build-Timestamp");
		if(buildTimestampString == null){
			throw new RuntimeException("The SkLearn2PMML package is not dated");
		}

		return Instant.parse(buildTimestampString);
	}

	static
	private Map<String, String> parseEnvironment(String string){
		Map<String, String> result = new LinkedHashMap<>();

		String[] lines = string.split("\n");
		for(String line : lines){
			int colon = line.indexOf(':');

			String key = (line.substring(0, colon)).trim();
			String value = (line.substring(colon + 1)).trim();

			result.put(key, value);
		}

		return result;
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