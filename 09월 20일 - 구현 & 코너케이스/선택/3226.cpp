#include<iostream>
#include<string>
using namespace std;

int calculateCharge(int h, int m,int time) {
	int end_h = h;
	int end_m = (m + time) % 60;

	//������ �ð� ���
	end_h += ((m + time) / 60) % 24;

	//����� �߰��� �ٲ�� ���
	if (h == 6 && end_h == 7) {
		return (time - end_m) * 5 + end_m * 10;
	}
	else if (h == 18 && end_h == 19) {
		return (time - end_m) * 10 + end_m * 5;
	}
	//�ٲ��� �ʴ� ���
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
		//getline(�Է� ��Ʈ��, string ��ü, ������); 
		//���� �����ڸ� �������� �ʰ� ���������, �ʿ��� ��� ������ �����ڸ� ���� ������ ���ڿ��� �Է¹޾� string ��ü�� ������ �� �־��!
		getline(cin, h, ':');
		cin >> m >> time;
		charge += calculateCharge(stoi(h), m, time);
	}
	cout << charge;
}