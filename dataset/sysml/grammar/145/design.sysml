package '15_08-Range Restriction' {
	private import ISQ::*;
	private import SI::*;
	private import '15_01-Constants'::'Mathematical Constants'::pi;
	part def HeadLightsTiltKnob {
		attribute headLightsTile : LightBeamTiltAngleValue[1];
	}
	attribute def LightBeamTiltAngleValue :> PlaneAngleValue {
		attribute angle: LightBeamTiltAngleValue :>> self {
		}
		assert constraint { angle >= 50 ['°'] and angle <= 80 ['°'] }
	}
}