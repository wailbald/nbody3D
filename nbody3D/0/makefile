all: nbodymieux.g nbodymieux2.g nbodymieux3.g nbody.g nbodymieuxfmadd.g nbodymieuxfmadd512.g nbodymieux.i nbodymieux2.i nbodymieux3.i nbody.i nbodymieuxfmadd.i nbodymieuxfmadd512.i

nbody.g: nbody.c
	gcc -march=native -mavx2 -Ofast -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbodymieux.g: nbodymieux.c
	gcc -march=native -mavx2 -Ofast -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbodymieux2.g: nbodymieux2.c
	gcc -march=native -mavx2 -Ofast -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbodymieux3.g: nbodymieux3.c
	gcc -march=native -mtune=native -mavx2 -Ofast -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbodymieuxfmadd.g: nbodymieuxfmadd.c
	gcc -march=native -mtune=native -mavx2 -Ofast -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbodymieuxfmadd512.g: nbodymieuxfmadd512.c
	gcc -march=native -mtune=native -mavx2 -Ofast -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp

nbody.i: nbody.c
	icc -xhost -Ofast -qopt-report $< -o $@ -qmkl -qopenmp

nbodymieux.i: nbodymieux.c
	icc -xhost -Ofast -qopt-report $< -o $@ -qmkl -qopenmp

nbodymieux2.i: nbodymieux2.c
	icc -xhost -Ofast -qopt-report $< -o $@ -qmkl -qopenmp

nbodymieux3.i: nbodymieux3.c
	icc -xhost -Ofast -qopt-report $< -o $@ -qmkl -qopenmp
	
nbodymieuxfmadd.i: nbodymieuxfmadd.c
	icc -xhost -Ofast -qopt-report $< -o $@ -qmkl -qopenmp

nbodymieuxfmadd512.i: nbodymieuxfmadd512.c
	icc -xhost -Ofast -qopt-report $< -o $@ -qmkl -qopenmp

clean:
	rm -Rf *~ nbody.g nbody.i *.optrpt *.g

