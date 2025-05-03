package 'VehicleSafetyandSecurityComplianceCatalog' {
	private import ScalarValues::*;
	private import AnnotationDefinitions::*;
	private import PartsTree::*;
	package AnnotationDefinitions {
		metadata def Safety {
			attribute isMandatory : Boolean;
		}
		metadata def Security;
	}
	package PartsTree {
		part vehicle {
			part interior {
				part alarm {@Security;}
				part seatBelt[2] {@Safety{isMandatory = true;}}
				part frontSeat[2];
				part driverAirBag {@Safety{isMandatory = false;}}
			}
			part bodyAssy {
				part body;
				part bumper {@Safety{isMandatory = true;}}
				part keylessEntry {@Security;}
			}
			part wheelAssy {
				part wheel[2];
				part antilockBrakes[2] {@Safety{isMandatory = false;}}
			}
		}
	}
	package 'Safety Features' {	
		private import vehicle::**[@Safety];
	}
	package 'Security Features' {		
		private import vehicle::**[@Security];
	}
	package 'Safety & Security Features' {		 
		private import vehicle::**[@Safety or @Security];
	}
	package 'Mandatory Saftey Features' {
		private import vehicle::**[@Safety and Safety::isMandatory];
	}
}
