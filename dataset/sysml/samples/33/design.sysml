package 'CameraMessagingSystem' {
	item def Scene;
	item def Image;
	item def Picture;
	attribute def Show {
		item picture : Picture;
	}
	action def Focus { in item scene : Scene; out item image : Image; }
	action def Shoot { in item image : Image; out item picture : Picture; }
	action def TakePicture;
	part screen {
		port displayPort;
	}
	part camera {
		port viewPort;
		port displayPort;
		action takePicture : TakePicture {
			action trigger accept scene : Scene via viewPort;
			then action focus : Focus {
				in item scene = trigger.scene;
				out item image;
			}
			flow from focus.image to shoot.image;
			then action shoot : Shoot {
				in item image; 
				out item picture;
			}
			then send Show(shoot.picture) via displayPort;
		}
	}
}