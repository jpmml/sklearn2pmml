/*
 * Copyright (c) 2024 Villu Ruusmann
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

import java.util.Objects;

import com.beust.jcommander.IStringConverter;
import org.dmg.pmml.Version;

public class VersionConverter implements IStringConverter<Version> {

	@Override
	public Version convert(String string){
		Version[] versions = Version.values();

		for(Version version : versions){

			if(!version.isStandard()){
				continue;
			} // End if

			if(Objects.equals(version.getNamespaceURI(), string) || Objects.equals(version.getVersion(), string)){
				return version;
			}
		}

		throw new IllegalArgumentException(string);
	}
}