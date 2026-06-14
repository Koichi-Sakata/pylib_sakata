#ifndef _HEAD_CTRLPRM_
#define _HEAD_CTRLPRM_

#define AXIS_NUM	6	    // Number of control axis
#define NF_NUM		1	    // Number of notch filter
#define RF_NUM		6	    // Number of resonant filter

typedef struct {
	double	dA[2];
	double	dB[2];
	double	dInPre;
	double	dOutPre;
} TF1_INF;					// 1st order TF information

typedef struct {
	double	dA[3];
	double	dB[3];
	double	dInPre[2];
	double	dOutPre[2];
} TF2_INF;					// 2nd order TF information

typedef struct {
	double	dA[4];
	double	dB[4];
	double	dInPre[3];
	double	dOutPre[3];
} TF3_INF;					// 3rd order TF information

extern TF2_INF	gstModelInf[AXIS_NUM];
extern TF2_INF	gstPIDInf[AXIS_NUM];
extern TF1_INF	gstPDInf[AXIS_NUM];
extern TF1_INF	gstPIInf[AXIS_NUM];
extern TF2_INF	gstNFInf[AXIS_NUM][NF_NUM];
extern TF2_INF	gstRFInf[AXIS_NUM][RF_NUM];
extern TF2_INF	gstDOBfbuInf[AXIS_NUM];
extern TF2_INF	gstDOBfbyInf[AXIS_NUM];
extern TF2_INF	gstDOBestuInf[AXIS_NUM];
extern TF2_INF	gstDOBestyInf[AXIS_NUM];
extern TF3_INF	gstZPETInf[AXIS_NUM];
extern TF2_INF	gstImpInf[AXIS_NUM];
extern TF1_INF	gstHapInf[AXIS_NUM];

#endif
