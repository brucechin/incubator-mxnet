#pragma once

#include<iostream>
#include<vector>

using namespace std;

template<class T>
class Share {
public:
    bool flag[8];
    T mean ;
    bool meanReady = false;
    int nDev=8;
    T data[8];
    Share(){
        memset(flag,false,sizeof(bool));
        meanReady = false;
    }
	bool Push(T input,int index) {

		if (flag[index] == false) {
			data[index] = input;
			flag[index] = true;
			return true;
		}
		else
			return false;
	}

	bool MeanReady() {
		if (!meanReady) {
           	     if(!flag[0]){ 
			std::cout<<"not ready 0"<<std::endl;
			return false;
			}
		     for (int i = 1; i < nDev; i++) {
			if (!flag[i]){
			    std::cout<<"not ready "<<i<<std::endl;
			    return false;
			}
			else data[0] += data[i];
		     }
		     mean = data[0];
	           
		     meanReady = true;
		     return true;
		}else 
		     return true;
	}

	void ResetMean() {
		for (int i = 0; i < nDev; i++) {
			if (flag[i]) return;
		}
		//std::cout<<"reset mean!"<<std::endl;
		meanReady = false;
	}

	T Pop(int index) {
		while(!MeanReady());//deadlock may occur 
        	flag[index] = false;
		T tmp = mean;
		ResetMean();
		return tmp;    
	}

};
