package 'VehicleDefinition' {
	part def Vehicle;	
	part def Engine;	
	part def Cylinder;
	part vehicle : Vehicle {
		part eng : Engine {
			part cyl : Cylinder[4..6];
		}
	}
	part smallVehicle :> vehicle {
		part redefines eng {
			part redefines cyl[4];
		}
	}
	part bigVehicle :> vehicle {
		part redefines eng {
			part redefines cyl[6];
		}
	}
}