package 'LayeredSystemDependency' {
	package 'Application Layer';
	package 'Service Layer';
	package 'Data Layer';
	dependency Use from 'Application Layer' to 'Service Layer';
	dependency from 'Service Layer' to 'Data Layer';
	attribute x;
	attribute y;
	attribute z;
	dependency z to x, y;
}