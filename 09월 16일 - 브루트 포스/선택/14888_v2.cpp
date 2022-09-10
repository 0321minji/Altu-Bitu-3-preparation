#include<iostream>
using namespace std;

int n;
int number[11];
int opt[4];
//�������� �־��� ���� ����� ������ ó�� MAX���� MIN�� �������ּ���
int MAX = -1000000001;
int MIN = 1000000001;

void Solution(int result, int idx) {
	if (idx == n) {
		if (result > MAX) {
			MAX = result;
		}
		if (result < MIN) {
			MIN = result;
		}
		return;
	}

	for (int i = 0; i < 4; i++) {
		if (opt[i]) {
			opt[i]--;
			if (i == 0) {
				Solution(result + number[idx], idx + 1);
			}
			else if (i == 1) {
				Solution(result - number[idx], idx + 1);

			}
			else if (i == 2) {
				Solution(result * number[idx], idx + 1);

			}
			else {
				Solution(result / number[idx], idx + 1);
			}
			opt[i]++; //��Ϳ��� return���� �� ����� ������ �ٽ� ��������
		}
	}
	return;
}

/*
* Solution �� ���� �ϳ��� �����ڸ� �����ϰ� ��� 
*-> �Լ��� �Ű������� �� �������� ������ �����ڷ� ����� ���(result)�� ���� ���꿡 ����� ������ �ε��� ���� idx�� �ѱ�
*  base�� ���� ���꿡 ����� ������ �ε����� n�� �� �� ���ڰ� ���� ��->result�� MAX, MIN ���ؼ� �� ����
*/

int main() {
	cin >> n;
	for (int i = 0; i < n; i++) {
		cin >> number[i];
	}

	cin >> opt[0] >> opt[1] >> opt[2] >> opt[3];

	Solution(number[0], 1);
	cout << MAX << '\n' << MIN;

	return 0;
}