#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

using namespace std;

int main(int argc, char *argv[])
{


    int nmole = 1;
    int atomchain = 100;       
//////////Reading sequence //////
	vector<double> sequence;
	ifstream infile1;
	infile1.open("sequence.txt");
	for (int i = 0; i < atomchain; i++)
	{
	string sstr;
	getline(infile1,sstr);
        stringstream typestring(sstr);
	int temptype;
	typestring >> temptype;
	sequence.push_back(temptype);	
	}
	infile1.close();
///////////////////////////////

	ifstream infile;
	infile.open("polymer1.data");
        ofstream outfile;
        outfile.open("polymer2.data");

        string str;    
	//Skiping lines 
	for(int i = 0; i < 2; i++)
        {
        getline(infile,str);
        }
	
    getline(infile,str);
    int Natom;
    stringstream atoms(str);
    atoms >> Natom;

    getline(infile,str);
    int Nbond;
    stringstream bonds(str);
    bonds >> Nbond;

    getline(infile,str);
    int NatomType;
    stringstream atomtypes(str);
    atomtypes >> NatomType;
   
    getline(infile,str);
    int NbondType;
    stringstream bondtypes(str);
    bondtypes >> NbondType;

    getline(infile,str);

    
	outfile << "LAMMPS data file" << endl;
        outfile << "    " << endl;
        outfile << Natom <<" atoms" << endl;
        outfile << Nbond <<" bonds" << endl;
        outfile << NatomType <<" atom types" << endl;
        outfile << NbondType <<" bond types" << endl;
        outfile << "    " << endl;

	/// Xbondary
	    getline(infile,str);
	    double xhi,xlo;
	    stringstream pbx(str);
	    pbx >> xlo >> xhi;
	    outfile << xlo << " " << xhi << " " << "xlo xhi" << endl;

	//Ybondary
	getline(infile,str);
        double yhi,ylo;
        stringstream pby(str);
        pby >> ylo >> yhi;
        outfile << ylo << " " << yhi << " " << "ylo yhi" << endl;

	
	//Zbondary
        getline(infile,str);
        double zhi,zlo;
        stringstream pbz(str);
        pbz >> zlo >> zhi;
        outfile << zlo << " " << zhi << " " << "zlo zhi" << endl;

        for (int i=0 ; i <3; i++)
        {
	    getline(infile,str);
        }

	outfile << " " << endl;
        outfile << "Masses" << endl;
        outfile << " " << endl;
	    
        vector<double> Mass;
        for (int i=0; i < NatomType; i++)
        {
        string str;
        getline(infile,str);
        stringstream atommass(str);
        double tempID, mass;
        atommass >> tempID >> mass;
        Mass.push_back(mass);
        }

        for (int i=0 ; i <3; i++)
        {   
        getline(infile,str);
        }   

        for (int i=0; i < NatomType; i++)
        {
        outfile << i+1 << "  " << Mass[i]  << endl;
        }

	outfile << " " << endl;
	outfile << "Atoms" << endl;
        outfile << " " << endl;

        string str1;
	int atomid = 0;
	
	double ID[50000], XX[50000], YY[50000], ZZ[50000], NNx[50000], NNy[50000], NNz[50000], MolType[50000];
	
	for (int countAtom = 0; countAtom < Natom; countAtom++)
	{
	getline(infile,str1);
    	double xtemp,ytemp,ztemp,charge;
    	charge = 0.0;
	int idtemp,moltemp,typetemp,nx,ny,nz;
	stringstream ss(str1);
	ss >> idtemp >> moltemp >> typetemp >>  xtemp >> ytemp >> ztemp >> nx >> ny >> nz;
        
	if(typetemp == 1 || typetemp == 2)
	{
		ID[idtemp]=idtemp;
		XX[idtemp]=xtemp;
        	YY[idtemp]=ytemp;
		ZZ[idtemp]=ztemp;
        	NNx[idtemp]=nx;
        	NNy[idtemp]=ny;
        	NNz[idtemp]=nz;
        	MolType[idtemp]=moltemp;
		}
		else
		{
		outfile << idtemp << "  "  << moltemp << "  " << typetemp <<  "  " << xtemp << "  " << ytemp  << "  " << ztemp << "  " << nx << "  " << ny << "  " << nz <<"\n";
		}
	}

/////////// writting compatabilizer ///////
//////////////////////////////////////////
	for (int i =0; i < nmole; i++)
	{
		for (int j =0; j < atomchain; j++)
        	{
		int index = ((i*atomchain) +1) +j;
                outfile << ID[index] << "  "  << MolType[index] << "  " << sequence[j] <<  "  " << XX[index] << "  " << YY[index]  << "  " << ZZ[index] << "  " << NNx[index] << "  " << NNy[index] << "  " << NNz[index] <<"\n";

        	}	
	
	}
	
/////Velocity ///////
////////////////////
/*
    for (int i=0; i < 3; i++)
    {
    getline(infile,str1);
    }

    outfile << " " << endl;
    outfile << "Velocities" << endl;
    outfile << " " << endl;

    for (int i=0; i < Natom; i++)
    {
    getline(infile,str1);
    outfile << str1 << "\n";
    }
*/
////Bonds ///
////////////
    for (int i=0; i < 3; i++)
    {
    getline(infile,str1);
    }

    outfile << " " << endl;
    outfile << "Bonds" << endl;
    outfile << " " << endl;

    for (int i=0; i < Nbond; i++)
    {
    getline(infile,str1);
    outfile << str1 << "\n";
    }
    
    infile.close();
    outfile.close();
}
