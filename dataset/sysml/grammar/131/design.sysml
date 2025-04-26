package '13a-Model Containment' {
	private import '2a-Parts Interconnection'::*;
	private import '8-Requirements'::*;
	requirement BodyAndInteriorRequirements {
		public import MassLimitationRequirement; 
	}
	requirement PowerTrainRequirements;
	package 'Vehicle Model' {
		package 'Vehicle1-Configuration' {			
			alias 'Sport Sedan' for vehicle1_c1;
			private import 'vehicle1_c1 Specification Context'::'vehicle1-c1 Specification';		
		}
		package 'Vehicle Reference Model' {
			private import VehicleA;			
			private import VehicleSubsystems;
		}
		package VehicleSubsystems {
			private import 'Body&Interior';
			private import 'PowerTrain';
		}
		package 'Body&Interior' {
			private import BodyAndInteriorRequirements;			
		}
		package PowerTrain {
			private import Engine;
			private import Transmission;
			private import PowerTrainRequirements;			
		}
	}
}