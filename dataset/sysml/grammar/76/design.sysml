package 'VehicleMassVerificationTestSystem' {
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
	part def MassVerificationSystem;
	part def Scale;
	part vehicleTestConfig : Vehicle {
		// ...
	}
	verification vehicleMassTest : VehicleMassTest {
		subject testVehicle :> vehicleTestConfig;
	}
	part massVerificationSystem : MassVerificationSystem {
		perform vehicleMassTest;	
		part scale : Scale {
			perform vehicleMassTest.collectData {
				in part :>> testVehicle;
				// In reality, this would be some more involved process.
				measurement = testVehicle.mass;
				out :>> massMeasured = measurement;
			}
		}
	}		
	individual def TestSystem :> MassVerificationSystem;
	individual def TestVehicle1 :> Vehicle;
	individual def TestVehicle2 :> Vehicle;
	individual testSystem : TestSystem :> massVerificationSystem {
		timeslice test1 {
			perform action :>> vehicleMassTest {
				in individual :>> testVehicle : TestVehicle1 {
					:>> mass = 2500[SI::kg];
				}
			}
		}
		then timeslice test2 {
			perform action :>> vehicleMassTest {
				in individual :>> testVehicle : TestVehicle2 {
					:>> mass = 3000[SI::kg];
				}
			}
		}
	}
}