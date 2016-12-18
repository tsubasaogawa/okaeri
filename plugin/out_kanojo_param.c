#include <stdio.h>
#include <string.h>
#include "plugin_defs.h"

#define PLUGIN_TITLE "A Feature Extractor for Okaeri Kanojo"

/** 
 */
int initialize() {
  return 0;
}

/** 
 */
int get_plugin_info(int opcode, char *buf, int buflen) {
  switch(opcode) {
  case 0:
    /* plugin description string */
    strncpy(buf, PLUGIN_TITLE, buflen);
    break;
  }

  return 0;
}

/* ready to speech */
void func_ready(Recog *recog, void *dummy) {
}

/* beginning of the recognition */
void func_begin(Recog *recog, void *dummy) {
}

/* end of the recognition */
void func_end(Recog *recog, void *dummy) {
}

/** 
 */
void adin_postprocess(SP16 *buf, int len) {
  printf("adin_postprocess: len=%d\n", len);
}

/* end of file */
