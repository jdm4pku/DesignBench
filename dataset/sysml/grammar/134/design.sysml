package 'VehicleSafetyandSecurityComponentRegistry' {
	part vehicle1_c1 {
		part interior {
			part alarm;
			part seatBelt[2];
			part frontSeat[2];
			part driverAirBag;
		}
		part bodyAssy {
			part body;
			part bumper;
			part keylessEntry;
		}
	}
	package 'Safety Features' {
		private import vehicle1_c1::interior::seatBelt;
		private import vehicle1_c1::interior::driverAirBag;
		private import vehicle1_c1::bodyAssy::bumper;		
	}
	package 'Security Features' {
		private import vehicle1_c1::interior::alarm;
		private import vehicle1_c1::bodyAssy::keylessEntry;
	}
	package 'Safety & Security Features' {
		private import 'Safety Features'::*;
		private import 'Security Features'::*;
	}
}
