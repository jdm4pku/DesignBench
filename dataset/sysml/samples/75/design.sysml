package 'VehicleMassComplianceVerification' {
	part def Vehicle {
		attribute mass :> ISQ::mass;
	}
	requirement vehicleMassRequirement {
		subject vehicle : Vehicle;
		in massActual :> ISQ::mass = vehicle.mass;
		doc /* The vehicle mass shall be less than or equal to 2500 kg. */
		
		require constraint { massActual <= 2500[SI::kg] }
	}	
	verification def VehicleMassTest {
		private import VerificationCases::*;
		subject testVehicle : Vehicle;
		objective vehicleMassVerificationObjective {
			// The subject of the verify is automatically bound to 'testVehicle' here.
			verify vehicleMassRequirement;
		}
		action collectData {
			in part testVehicle : Vehicle = VehicleMassTest::testVehicle;
			out massMeasured :> ISQ::mass;
		}
		action processData {
			in massMeasured :> ISQ::mass = collectData.massMeasured;
			out massProcessed :> ISQ::mass;
		}
		action evaluateData {
			in massProcessed :> ISQ::mass = processData.massProcessed;
			out verdict : VerdictKind = 
				// Check that 'testVehicle' statisfies 'vehicleMassRequirement' if its mass equals 'massProcessed'.
				PassIf(vehicleMassRequirement(vehicle = testVehicle(mass = massProcessed)));
		}
		return verdict : VerdictKind = evaluateData.verdict;
	}
}