package 'VehicleDefinition' {
	part def Vehicle {
		part eng : Engine;
	}
	part def Engine {
		part cyl : Cylinder[4..6];
	}
	part def Cylinder;	
	part smallVehicle : Vehicle {
		part redefines eng {
			part redefines cyl[4];
		}
	}
	part bigVehicle : Vehicle {
		part redefines eng {
			part redefines cyl[6];
		}
	}
}