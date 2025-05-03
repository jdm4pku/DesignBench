package 'VehicleRequirementsSpecialization' {
	private import ISQ::*;
	private import SI::*;
	requirement def MassLimitationRequirement {
		doc /* The actual mass shall be less than or equal to the required mass. */
		attribute massActual: MassValue;
		attribute massReqd: MassValue;
		require constraint { massActual <= massReqd }
	}
	part def Vehicle {
		attribute dryMass: MassValue;
		attribute fuelMass: MassValue;
		attribute fuelFullMass: MassValue;
	}
	requirement def <'1'> VehicleMassLimitationRequirement :> MassLimitationRequirement {
		doc /* The total mass of a vehicle shall be less than or equal to the required mass. */
		subject vehicle : Vehicle;
		attribute redefines massActual = vehicle.dryMass + vehicle.fuelMass;
		assume constraint { vehicle.fuelMass > 0[kg] }
	}
	port def ClutchPort;
	action def GenerateTorque;
	requirement def <'2'> DrivePowerInterface {
		doc /* The engine shall transfer its generated torque to the transmission via the clutch interface. */
		subject clutchPort: ClutchPort;
	}	
	requirement def <'3'> TorqueGeneration {
		doc /* The engine shall generate torque as a function of RPM as shown in Table 1. */
		subject generateTorque: GenerateTorque;
	}
	requirement <'1.1'> fullVehicleMassLimit : VehicleMassLimitationRequirement {
		subject vehicle : Vehicle;
		attribute :>> massReqd = 2000[kg];
		assume constraint {
			doc /* Full tank is full. */
			vehicle.fuelMass == vehicle.fuelFullMass
		}
	}
	requirement <'1.2'> emptyVehicleMassLimit : VehicleMassLimitationRequirement {
		subject vehicle : Vehicle;
		attribute :>> massReqd = 1500[kg];	
		assume constraint {
			doc /* Full tank is empty. */
			vehicle.fuelMass == 0[kg]
		}
	}
}