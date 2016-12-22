CC       = gcc
JDIR     = J:/MinGW/msys/1.0/home/gomashio/julius
IDIR1    = $(JDIR)/libjulius/include
IDIR2    = $(JDIR)/libsent/include
LDIR1    = $(JDIR)/libjulius
LDIR2    = $(JDIR)/libsent
LIBS     = `$(JDIR)/libjulius/libjulius-config --libs` `$(JDIR)/libsent/libsent-config --libs` 
DEFINES  = `$(JDIR)/libjulius/libjulius-config --cflags` `$(JDIR)/libsent/libsent-config --cflags` 

all : out_okaeri_param.jpi

out_okaeri_param.jpi : out_okaeri_param.o
	$(CC) -shared -o out_okaeri_param.jpi out_okaeri_param.o -L$(LDIR1) -L$(LDIR2) $(LIBS)

out_okaeri_param.o : out_okaeri_param.c
	$(CC) -c out_okaeri_param.c -Wall -I$(IDIR1) -I$(IDIR2) $(DEFINES)

