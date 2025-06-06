package 'VehicleDynamicsTemporalSimulation' {
	private import ScalarValues::Real;
	private import Time::DateTime;
	private import ISQ::*;
	package 'Part Definitions' {	
		part def 'Temporal-Spatial Reference' {
			attribute referenceTime : DateTime;
			attribute referenceCoordinateSystem;
		}
		part def VehicleRoadContext {
			attribute t : TimeValue;
		}
		part def VehicleA {
			attribute mass : MassValue;
			attribute position : Real;
			attribute velocity : Real;
			attribute acceleration : Real;
			exhibit state vehicleStates {
				entry; then on;
				state on;
				then off;
				state off;
			}
		}
		part def Road {
			attribute angle : Real;
			attribute surfaceFriction : Real;
		}
	}
	package 'Individual Definitions' {
		private import 'Part Definitions'::*;
		individual def 'Temporal-Spatial Reference_ID1' :> 'Temporal-Spatial Reference';
		individual def VehicleRoadContext_ID1 :> VehicleRoadContext;
		individual def VehicleA_ID1 :> VehicleA;
		individual def Road_ID1 :> Road;
	}
	package Values {	
		attribute t0 : TimeValue;
		attribute t1 : TimeValue;
		attribute tn : TimeValue;
		attribute m : MassValue;
		attribute p0 : Real;
		attribute p1 : Real;
		attribute pn : Real;
		attribute v0 : Real;
		attribute v1 : Real;
		attribute vn : Real;
		attribute a0 : Real;
		attribute a1 : Real;
		attribute an : Real;
		attribute theta0 : Real;
		attribute theta1 : Real;
		attribute thetan : Real;
		attribute sf0 : Real;
		attribute sf1 : Real;
		attribute sfn : Real;
	}
	package 'Individuals and Snapshots' {
		private import 'Individual Definitions'::*;
		private import Values::*;	
		individual reference : 'Temporal-Spatial Reference_ID1' {
			snapshot context_t0 : VehicleRoadContext_ID1 {
				:>> t = t0 {
				}
				snapshot vehicle_ID1_t0 : VehicleA_ID1 {
					:>> mass = m;
					:>> position = p0;
					:>> velocity = v0;
					:>> acceleration = a0;
					exhibit vehicleStates.on {
					}
				}
				snapshot road_ID1_t0 : Road_ID1 {
					:>> angle = theta0;
					:>> surfaceFriction = sf0;
				}
			}
			snapshot context_t1 : VehicleRoadContext_ID1 {
				:>> t = t1;	
				snapshot vehicle_ID1_t1 : VehicleA_ID1 {
					:>> mass = m;
					:>> position = p1;
					:>> velocity = v1;
					:>> acceleration = a1;
					exhibit vehicleStates.on;
				}
				snapshot road_ID1_t1 : Road_ID1 {
					:>> angle = theta1;
					:>> surfaceFriction = sf1;
				}
			}
			snapshot context_tn : VehicleRoadContext_ID1 {
				:>> t = tn;
				snapshot vehicle_ID1_tn : VehicleA_ID1 {
					:>> mass = m;
					:>> position = pn;
					:>> velocity = vn;
					:>> acceleration = an;
					exhibit vehicleStates.off;
				}
				snapshot road_ID1_tn : Road_ID1 {
					:>> angle = theta1;
					:>> surfaceFriction = sfn;
				}
			}
		}
	}
}