package 'PhotoCaptureWorkflow' {
	part def Scene;
	part def Image;
	part def Picture;
	action def Focus { in scene : Scene; out image : Image; }
	action def Shoot { in image: Image; out picture : Picture; }	
	action def TakePicture { in scene : Scene; out picture : Picture; }
	action takePicture : TakePicture {
		in item scene;
		out item picture;
		action focus : Focus {
			in item scene = takePicture::scene; 
			out item image;
		}
		flow from focus.image to shoot.image;
		action shoot : Shoot {
			in item; 
			out item picture = takePicture::picture;
		}
	}
	part def Camera;
	part def AutoFocus;
	part def Imager;
	part camera : Camera {
		perform action takePhoto[*] ordered 
			references takePicture;
		part f : AutoFocus {
			perform takePhoto.focus;			
		}
		part i : Imager {
			perform takePhoto.shoot;
		}		
	}
}