package 'AutomotiveSystemArchitectureSafetyViews' {
	private import Views::*;
	private import ScalarValues::Boolean;
	part def 'Systems Engineer';
	part def 'IV&V';
	concern 'system breakdown' {
		doc /* 
		 * To ensure that a system covers all its required capabilities,
		 * it is necessary to understand how it is broken down into
		 * subsystems and components that provide those capabilities.
		 */
		stakeholder se : 'Systems Engineer';
		stakeholder ivv : 'IV&V';
	}
	concern 'modularity' {
		doc /*
		 * There should be well defined interfaces between the parts of
		 * a system that allow each part to be understood individually,
		 * as well as being part of the whole system.
		 */	 
		 stakeholder se : 'Systems Engineer';
	}
	viewpoint 'system structure perspective' {		
		frame 'system breakdown';
		frame 'modularity';	
		require constraint {
			doc /*
			 * A system structure view shall show the hierarchical 
			 * part decomposition of a system, starting with a 
			 * specified root part.
			 */
		}
	}
	metadata def Safety {
		attribute isMandatory : Boolean;
	}
	part vehicle {
		part interior {
			part alarm;
			part seatBelt[2] {@Safety{isMandatory = true;}}
			part frontSeat[2];
			part driverAirBag {@Safety{isMandatory = false;}}
		}
		part bodyAssy {
			part body;
			part bumper {@Safety{isMandatory = true;}}
			part keylessEntry;
		}
		part wheelAssy {
			part wheel[2];
			part antilockBrakes[2] {@Safety{isMandatory = false;}}
		}
	}
	package 'Safety Features' {
		/* Parts that contribute to safety. */		
		private import vehicle::**[@Safety];
	}
	package 'Mandatory Safety Features' {
		/* Parts that contribute to safety AND are mandatory. */
		private import vehicle::**[@Safety and Safety::isMandatory];
	}
	view def 'Part Structure View' {
		satisfy 'system structure perspective';		
		filter @SysML::PartUsage;
	}
	view 'vehicle structure view' : 'Part Structure View' {
		expose vehicle::**;
		render asTreeDiagram;
	}
	rendering asTextualNotationTable :> asElementTable {
		view :>> columnView[1] {
			render asTextualNotation;
		}
	}
	view 'vehicle tabular views' {
		view 'safety features view' : 'Part Structure View' {
			expose vehicle::**[@Safety];
			render asTextualNotationTable;
		}
		view 'non-safety features view' : 'Part Structure View' {
			expose vehicle::**[not (@Safety)];
			render asTextualNotationTable;
		}
	}
}