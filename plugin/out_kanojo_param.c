#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "julius/juliuslib.h"
#include "plugin_defs.h"

#define PLUGIN_TITLE "A Feature Extractor for Okaeri Kanojo"
#define MAX_RESULT_LEN 2048

/* プロトタイプコール */
int startup(void *data);
int get_plugin_info(int opcode, char *buf, int buflen);
void output_result(Recog *recog, void *dummy);
// void func_end(Recog *recog, void *dummy);

/* スタートアップ関数. コールバックを登録する */
int startup(void *data) {
  Recog *recog = data;
  callback_add(recog, CALLBACK_RESULT, output_result, NULL);  
  // callback_add(recog, CALLBACK_EVENT_RECOGNITION_END, func_end, NULL);
  return 0;
}

/* プラグイン情報を定義する (必ず必要) */
int get_plugin_info(int opcode, char *buf, int buflen) {
  switch(opcode) {
  case 0:
    /* plugin description string */
    strncpy(buf, PLUGIN_TITLE, buflen);
    break;
  }

  return 0;
}

void output_result(Recog *recog, void *dummy) {
  int i, j;
  RecogProcess *r;
  WORD_INFO *winfo;
  WORD_ID *wid;
  Sentence *sent;
  int samplenum = recog->lmlist->am->mfcc->param->samplenum;

  /* get the mfcc & dmfcc values. */
  float **mfcc = recog->lmlist->am->mfcc->param->parvec;
  int mfcclen = recog->lmlist->am->mfcc->param->veclen;
  float *avg_mfcc;
  char result[MAX_RESULT_LEN];

  /* 認識結果取得 */
  for(r=recog->process_list; r; r=r->next) {
    if(! r->live || r->result.status < 0) continue;

    /* 今回はベストの候補のみ取得する */
    winfo = r->lm->winfo;
    sent = &(r->result.sent[0]);
    wid = sent->word;
    for(i=0; i<sent->word_num; i++) 
      printf("yeah: %s\n", winfo->woutput[wid[i]]);
   // strcpy(result, winfo->woutput[sent->word[0]]);
  }

  /* メモリ割り当て */
  avg_mfcc = (float *) malloc(sizeof(float) * mfcclen);
  for(i=0; i<mfcclen; i++) avg_mfcc[i] = 0.0;

  /* mfcc の瞬間値を平均化する */
  for(i=0; i<samplenum; i++) {
    for(j=0; j<mfcclen; j++) {
      avg_mfcc[j] += mfcc[i][j];
    }
  }
  for(i=0; i<mfcclen; i++) {
    avg_mfcc[i] /= samplenum;
    printf("%.2f ", avg_mfcc[i]);
  }
  printf("\n");
  free(avg_mfcc);
}

/* end of file */
