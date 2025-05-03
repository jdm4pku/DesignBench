package 'VehicleMassProperty' {
	private import ISQ::*;
	private import SI::*;
	private import NumericalFunctions::*;
	package VehicleDesignModel {
		part def Vehicle {
			mass : MassValue;
		}
		part vehicle {
			:>> mass : MassValue = sum((
				vehicle.engine.mass, 
				vehicle.transmission.mass, 
				vehicle.frontAxleAssembly.mass, 
				vehicle.rearAxleAssembly.mass
			));
			part engine {
				mass : MassValue;
			}
			part transmission {
			    mass : MassValue;
			}
			part frontAxleAssembly {
				mass : MassValue;
			}
			part rearAxleAssembly {
				mass : MassValue;
			}
		}
	}
	package VehicleAnalysisModel {
		private import VehicleDesignModel::Vehicle;
		requirement def MassAnalysisObjective {
			subject mass : MassValue;
		}
		analysis def MassAnalysisCase {
			subject vehicle : Vehicle;
			objective : MassAnalysisObjective;
			vehicle.mass
		}
		analysis def AnalysisPlan {
			subject vehicle : Vehicle;			
			objective {
			}
			analysis massAnalysisCase : MassAnalysisCase {
			 	return mass; 
			 }
		}
		part massAnalysisContext {
			analysis analysisPlan : AnalysisPlan {
				subject vehicle = VehicleDesignModel::vehicle;
			}
		}
	}
}