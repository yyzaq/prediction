#include <stdio.h>
#include <stdio.h>

void Delete(int a[]){
    int b,i,n[10],flag=0,j;
    scanf("%d",&b);
    for(i=0;i<10;i++){
        if (a[i]==b){
            n[flag]=i;
            flag++;
        }
    }
    for(i=0;i<flag;i++){
        for(j=n[flag];j<10;j++)
        a[j]=a[j+1];
    }
}
int main(){
    int a[10]={1,2,3,4,5,6,7,8,9,10};
    Delete(a);
    for(int i=0;i<10;i++){
        printf("%d ",a[i]);
    }
    return 0;
}