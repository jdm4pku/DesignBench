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
	part def Wheel;
	individual part def Vehicle_1 :> Vehicle {
		part leftFrontWheel : Wheel;
		part rightFrontWheel : Wheel;
	}
	individual part def Wheel_1 :> Wheel;
	individual part vehicle_1 : Vehicle_1 {
		snapshot part vehicle_1_t0 {
			snapshot leftFrontWheel_t0 : Wheel_1 :>> leftFrontWheel;
		}
		then snapshot part vehicle_1_t1 {
			snapshot rightFrontWheel_t1 : Wheel_1 :>> rightFrontWheel;
		}
	}
}