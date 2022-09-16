#include<iostream>
#include<string>
using namespace std;

int calculateCharge(int h, int m,int time) {
	int end_h = h;
	int end_m = (m + time) % 60;

	//끝나는 시간 계산
	end_h += ((m + time) / 60) % 24;

	//요금이 중간에 바뀌는 경우
	if (h == 6 && end_h == 7) {
		return (time - end_m) * 5 + end_m * 10;
	}
	else if (h == 18 && end_h == 19) {
		return (time - end_m) * 10 + end_m * 5;
	}
	//바뀌지 않는 경우
	else {
		if (6 < h && h < 19) {
			return 10 * time;
		}
		else {
			return 5 * time;
		}
	}
}


int main() {
	int n, m, time;
	int charge = 0;
	string h;
	cin >> n;
	while(n--) {
		//getline(입력 스트림, string 객체, 구분자); 
		//보통 구분자를 지정하지 않고 사용하지만, 필요한 경우 지정한 구분자를 만날 때까지 문자열을 입력받아 string 객체에 저장할 수 있어요!
		getline(cin, h, ':');
		cin >> m >> time;
		charge += calculateCharge(stoi(h), m, time);
	}
	cout << charge;
}