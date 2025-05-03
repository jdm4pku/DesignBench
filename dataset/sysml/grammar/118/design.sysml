package 'AutomotiveSpecificationSystem' {
	private import ScalarValues::Real;
	private import ISQ::*;
	private import SI::*;
	public import 'Vehicle Usages'::*;
	public import 'Vehicle Requirements'::*;
	package 'Vehicle Definitions' {
		part def Vehicle {
			attribute mass: MassValue;
			attribute fuelLevel: Real;
			attribute fuelTankCapacity: Real;
		}
		part def Engine {
			port drivePwrPort: DrivePwrPort;
			perform action 'generate torque': 'Generate Torque';
		}
		part def Transmission {
			port clutchPort: ClutchPort;
		}
		port def DrivePwrPort;
		port def ClutchPort;
		interface def EngineToTransmissionInterface {
			end drivePwrPort: DrivePwrPort;
			end clutchPort: ClutchPort;
		}
		action def 'Generate Torque';
	}
	package 'Vehicle Usages' {
		public import 'Vehicle Definitions'::*;
		action 'provide power' {
			action 'generate torque' { /* ... */ }
			//...
		}
		part vehicle1_c1: Vehicle {
			attribute :>> mass = 2000 [kg];
			perform 'provide power';	
			part engine_v1: Engine {
				port :>> drivePwrPort;
				perform 'provide power'.'generate torque' :>> 'generate torque';
			}
			part transmission: Transmission {
				port :>> clutchPort;
			}
			interface engineToTransmission: EngineToTransmissionInterface
				connect engine_v1.drivePwrPort to transmission.clutchPort;
		}
		part vehicle1_c2: Vehicle {
			attribute :>> mass = 2500 [kg];
		}
	}
	package 'Vehicle Requirements' {	
		public import 'Vehicle Definitions'::*;
		requirement def <'1'> MassLimitationRequirement {
			attribute massActual: MassValue;
			attribute massReqd: MassValue;
			require constraint {
				 massActual <= massReqd 
			 }
		}
		requirement def <'2'> ReliabilityRequirement;
		requirement <'1.1'> vehicleMass1: MassLimitationRequirement {
			subject vehicle : Vehicle {
			}
			attribute :>> massActual: MassValue = vehicle.mass {
			}
			attribute :>> massReqd = 2000 [kg] {
			}
			assume constraint fuelConstraint {
				vehicle.fuelLevel >= vehicle.fuelTankCapacity
			}
		}
		requirement <'2.1'> vehicleMass2: MassLimitationRequirement {
			subject vehicle : Vehicle;
			attribute :>> massActual: MassValue = vehicle.mass;
			attribute :>> massReqd = 2500 [kg];
			assume constraint fuelConstraint {
				vehicle.fuelLevel == 0.0
			}
		}
		requirement <'2.2'> vehicleReliability2: ReliabilityRequirement {
			subject vehicle : Vehicle;
		}
		requirement <'3.1'> drivePowerInterface {
			subject drivePwrPort: DrivePwrPort;
		}
		requirement <'3.2'> torqueGeneration {
			subject generateTorque: 'Generate Torque';
		}
	}
	part 'vehicle1_c1 Specification Context' {
		private import 'vehicle1-c1 Specification'::*;
		private import 'engine-v1 Specification'::*;
		requirement 'vehicle1-c1 Specification' {		
			subject vehicle : Vehicle;
			requirement references vehicleMass1 {			
			}
		}
		requirement 'engine-v1 Specification' {
			subject engine : Engine;
			require torqueGeneration {
				in :>> generateTorque = engine.'generate torque';
			}
			require drivePowerInterface {
				in :>> drivePwrPort = engine.drivePwrPort; 
			}
		}
		satisfy 'vehicle1-c1 Specification' by vehicle1_c1 {
		}
		satisfy 'engine-v1 Specification' by vehicle1_c1.engine_v1;
	}
	part 'vehicle1_c2 Specification Context' {
		private import 'vehicle1-c2 Specification'::*;
		requirement 'vehicle1-c2 Specification' {
			subject vehicle : Vehicle;
			require vehicleMass2;
			require vehicleReliability2;
		}
		satisfy 'vehicle1-c2 Specification' by vehicle1_c2;
	}
}