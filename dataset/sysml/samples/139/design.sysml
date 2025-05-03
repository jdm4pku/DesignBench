package 'AutomotiveComponentRules' {
	private import ControlFunctions::forAll;
	private import SI::*;
	private import SI::*;
    private import USCustomaryUnits::*;
    part def Vehicle_1 {
        attribute mass: MassValue = 1200 [kg];
        attribute length: LengthValue = 4.82 [m];
        part leftFrontWheel : Wheel;
        part rightFrontWheel : Wheel;
    }
    part def Wheel {
    	attribute hubDiameter: LengthValue = 18 ['in'];
        attribute width: LengthValue = 245 [mm];
        attribute outerDiameter: LengthValue = (hubDiameter + 2 * tire.height) [mm] {
	        doc
	        /*
	         * This binds 'outDiameter' to the result of a computed attribute.
	         * There is no need to mark it as "derived".
	         */
        }
        part tire: Tire[1];
    }
    part def Tire {
    	attribute profileDepth: LengthValue default 6.0 [mm];
        constraint hasLegalProfileDepth {profileDepth >= 3.5 [mm]}
    	attribute height: LengthValue = 45 [mm];
    }
	constraint def DiscBrakeConstraint {
		in wheelAssy : WheelAssy[4];
		wheelAssy->forAll {in ref w: WheelAssy; 
			2 * w.discBrakeAssy.radius < w.wheel.outerDiameter
		}
	}
	constraint def DiscBrakeFitConstraint_Alt {
		in discBrakeAssy : DiscBrakeAssy[1];
		in wheel : Wheel[1];	
		2 * discBrakeAssy.radius < wheel.outerDiameter
	}
	part def Vehicle_2 {
		attribute mass : MassValue[1] = 1200 [kg];
		attribute length : LengthValue[1] = 4.82 [m];
		part wheelAssy : WheelAssy[4];
		constraint discBrakeConstraint : DiscBrakeConstraint {
			in wheelAssy = Vehicle_2::wheelAssy;
		}
	}
	part def WheelAssy {
		part wheel : Wheel[1];
		part discBrakeAssy : DiscBrakeAssy[1];
		assert constraint discBrakeFitConstraint_Alt: DiscBrakeFitConstraint_Alt {
			in discBrakeAssy = WheelAssy::discBrakeAssy;
			in wheel = WheelAssy::wheel;
		}
	}
	part def DiscBrakeAssy {
		attribute radius : LengthValue[1] = 95 [mm];
	}
}