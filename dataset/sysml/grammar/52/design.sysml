package 'VehicleComposition' {
	private import ScalarValues::*;
	part def Vehicle {
		attribute mass : Real;
		attribute status : VehicleStatus;
		part eng : Engine;
		ref part driver : Person;
	}
	attribute def VehicleStatus {
		attribute gearSetting : Integer;
		attribute acceleratorPosition : Real;
	}
	part def Engine;	
	part def Person;
	individual part def Vehicle_1 :> Vehicle {
		snapshot part vehicle_1_t0 {
			:>> mass = 2000.0;
			:>> status {
				:>> gearSetting = 0;
				:>> acceleratorPosition = 0.0;
			}
		}
		snapshot part vehicle_1_t1 {
			:>> mass = 1500.0;
			:>> status {
				:>> gearSetting = 2;
				:>> acceleratorPosition = 0.5;
			}
		}
		first vehicle_1_t0 then vehicle_1_t1;
	}
}