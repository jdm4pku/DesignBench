package 'VehicleMassComplianceVerification' {
	private import VerificationCases::*;
	private import Definitions::*;
	package Definitions {
		requirement def <'2'> MassRequirement {
			attribute massActual :> ISQ::mass;
			attribute massReqd :> ISQ::mass;
			require constraint { massActual <= massReqd }
		}
		part def Vehicle {
			attribute mass :> ISQ::mass;
		}
		part def MassVerificationSystem;
		part def Scale;
		part def TestOperator;
		individual def TestVehicle1 :> Vehicle;
		individual def TestVehicle2 :> Vehicle;
		individual def TestSystem :> MassVerificationSystem;
		verification def MassTest {
			objective massVerificationObjective {
				verify requirement massRequirement : MassRequirement;
			}
		}
	}
	package Usages {
		requirement <'2.1'> vehicleMassRequirement : MassRequirement {
			subject vehicle : Vehicle;
			:>> massActual = vehicle.mass;		
			:>> massReqd = 2500 [SI::kg];
		}
		part vehicle1_c2 : Vehicle {
		}
		verification vehicleMassTest : MassTest {
			subject testVehicle : Vehicle;
			objective vehicleMassVerificationObjective {
				verify vehicleMassRequirement :>> massRequirement;
			}
			action collectData {
				in part testVehicle : Vehicle = vehicleMassTest.testVehicle;
				out massMeasured :> ISQ::mass;
			}
			action processData {
				in massMeasured :> ISQ::mass = collectData.massMeasured;
				out massProcessed :> ISQ::mass;
			}
			action evaluateData {
				in massProcessed :> ISQ::mass = processData.massProcessed;
				out verdict : VerdictKind = 
					PassIf(vehicleMassRequirement(vehicle = testVehicle(mass = massProcessed)));
			}
			return verdict : VerdictKind = evaluateData.verdict;
		}
		part massVerificationSystem : MassVerificationSystem {
			perform vehicleMassTest {
				in part :>> testVehicle = vehicleUnderTest;
			}
			ref part vehicleUnderTest : Vehicle;
			part testOperator : TestOperator;
			part scale : Scale {
				perform vehicleMassTest.collectData {
					in part :>> testVehicle;
					measurement = testVehicle.mass;
					out :>> massMeasured = measurement;
				}
			}
		}
		individual testSystem : TestSystem :> massVerificationSystem {
			timeslice test1 {
				ref individual :>> vehicleUnderTest : TestVehicle1 :> vehicle1_c2 {
					:>> mass = 2500 [SI::kg];
				}
			}
			then timeslice test2 {
				ref individual :>> vehicleUnderTest : TestVehicle2 :> vehicle1_c2 {
					:>> mass = 2500 [SI::kg];
				}
			}
		}
	}
}