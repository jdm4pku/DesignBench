package 'VehicleTrailerDynamicCouplingSystem' {
	part def Vehicle;
	part def VehicleFrame;
	part def HitchBall;
	part def TrailerCoupler;
	part def Trailer;
	part def TrailerFrame;
	connection def TrailerHitch {
		end hitch : HitchBall;
		end coupler : TrailerCoupler;
	}
	part 'vehicle-trailer system' {
		part vehicle : Vehicle {
			part vehicleFrame : VehicleFrame {
				part hitch : HitchBall;
			}
		}
		connection trailerHitch : TrailerHitch[0..1]
			connect vehicle.vehicleFrame.hitch to trailer.trailerFrame.coupler;
		part trailer : Trailer {
			part trailerFrame : TrailerFrame {
				part coupler : TrailerCoupler;
			}
		}
		action {
			action 'connect trailer to vehicle'
				assign 'vehicle-trailer system'.trailerHitch := TrailerHitch();
			then action 'destroy connection of trailer to vehicle' : 
				OccurrenceFunctions::destroy {
				inout occ = 'vehicle-trailer system'.trailerHitch;
			}
			then action 'disconnect trailer from vehicle'
				assign 'vehicle-trailer system'.trailerHitch := null;
		}	
	}	
}