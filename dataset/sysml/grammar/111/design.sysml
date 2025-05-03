package 'VehicleAssemblyProcessModeling' {
	public import Definitions::*;
	package Definitions {
		item def VehicleAssembly;
		item def AssembledVehicle :> VehicleAssembly;
		part def Vehicle :> AssembledVehicle;		
		part def Transmission;
		part def Engine;		
	}
	package Usages {	
		part AssemblyLine {
			perform action 'assemble vehicle' {
				action 'assemble transmission into vehicle' {
					in item 'vehicle assy without transmission or engine' : VehicleAssembly;					
					in item transmission : Transmission {
					}
					out item 'vehicle assy without engine' : VehicleAssembly = 'vehicle assy without transmission or engine' {						
						part transmission : Transmission = 'assemble transmission into vehicle'.transmission {
						}
					}
				}
				flow 'assemble transmission into vehicle'.'vehicle assy without engine' 
				    to 'assemble engine into vehicle'.'vehicle assy without engine';
				action 'assemble engine into vehicle' {
					in item 'vehicle assy without engine' : VehicleAssembly {
						part transmission : Transmission;
					}
					in item engine : Engine;
					out item assembledVehicle : AssembledVehicle = 'vehicle assy without engine' {
						part engine : Engine = 'assemble engine into vehicle'.engine;
					}
				}
			}
			bind 'assemble vehicle'.'assemble engine into vehicle'.assembledVehicle = vehicle;
			part vehicle : Vehicle {
				part transmission: Transmission;
				part engine: Engine;
				perform action providePower;
			}
		}
	}
}