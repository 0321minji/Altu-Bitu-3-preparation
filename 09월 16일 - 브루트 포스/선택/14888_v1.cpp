#include<iostream>
#include<vector>
#include<cmath>
using namespace std;

vector<int> number(11,0);
vector<int> opt(4, 0); //�� ������ ���� ����
vector<int> opt_order; //�����ڸ� ������ ������� �����ϰ�(push_back) ��Ϳ��� return���� �� �ڿ������� ���Ҹ� �����ϱ�(pop_back) vector�� ���!
int n;

//�������� �־��� ���� ����� ������ ó�� MAX���� MIN�� �������ּ���
int MAX = -1000000001;
int MIN = 1000000001;

int arithmeticOpt(int n1, int n2, int opt) {
	switch (opt)
	{
		case 0:
			return n1 + n2;
			break;
		case 1: 
			return n1 - n2;
			break;
		case 2 : 
			return n1 * n2;
			break;
		case 3:
			return n1 / n2; //C���� ���� �����Ⱑ ��� �̷�������� �������ּ���!
			break;
		default:
			break;
	}
}

void  calc() {

	int result = number[0];
	for (int i = 1; i < n; i++) {
		result = arithmeticOpt(result, number[i], opt_order[i-1]);
	}

	if (result > MAX) {
		MAX = result;
	}
	if (result < MIN) {
		MIN = result;
	}
}

void Solution(int cnt) {

	if(cnt == n-1) {
		calc();
		return;
	}

	for (int i = 0; i < 4; i++) {
		if (opt[i]) {

			opt[i]--;
			opt_order.push_back(i);

			Solution(cnt+1);

			opt[i]++;
			opt_order.pop_back();
		}
	}

}

/*
* Solution �� ���� �ϳ��� �����ڸ� �����ϰ� ���
* base�� �������� ������ n-1�� �� -> calc �Լ��� ���� ������ ������� ����ؼ� MAX, MIN����
* arithmeticOpt�� �������� ������ ���� ������ִ� �Լ�
*/

int main(void) {

	cin >> n;
	
	for (int i = 0; i < n; i++) {
		cin >> number[i];
	}

	cin >> opt[0] >> opt[1] >> opt[2] >> opt[3];

	Solution(0);
	cout << MAX << '\n' << MIN;

	return 0;
}