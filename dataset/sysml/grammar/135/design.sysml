package 'SecurityMetadataExtensions' {
	private import 'User Defined Extensions'::*;
	package 'User Defined Extensions' {
		enum def ClassificationLevel {
			uncl;
			conf;
			secret;
		}
		metadata def Classified {
			ref :>> annotatedElement : SysML::PartUsage;
			attribute classificationLevel : ClassificationLevel[1];
		}
	}
	part part_X {
		metadata Classified {
			classificationLevel = ClassificationLevel::conf;
		}
	}
	part part_Y {
		@Classified {
			classificationLevel = ClassificationLevel::conf;
		}
	}
}