#include "TcPch.h"
#pragma hdrstop
#include "head_ctrlprm.h"


TF2_INF	gstModelInf[6] = {
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	}
};

TF2_INF	gstPIDInf[6] = {
	{
		{ 1.0, -1.938516817916602, 0.9385168179166019 },
		{ 2033.0855299532648, -4033.4546496552894, 2000.5560605716003 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.938516817916602, 0.9385168179166019 },
		{ 2033.0855299532648, -4033.4546496552894, 2000.5560605716003 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.938516817916602, 0.9385168179166019 },
		{ 2033.0855299532648, -4033.4546496552894, 2000.5560605716003 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.938516817916602, 0.9385168179166019 },
		{ 2033.0855299532648, -4033.4546496552894, 2000.5560605716003 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.938516817916602, 0.9385168179166019 },
		{ 2033.0855299532648, -4033.4546496552894, 2000.5560605716003 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.938516817916602, 0.9385168179166019 },
		{ 2033.0855299532648, -4033.4546496552894, 2000.5560605716003 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	}
};

TF1_INF	gstPDInf[6] = {
	{
		{ 1.0, -0.9822783063440266 },
		{ 116.81329322863975, -116.1371994627773 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -0.9822783063440266 },
		{ 116.81329322863975, -116.1371994627773 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -0.9822783063440266 },
		{ 116.81329322863975, -116.1371994627773 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -0.9822783063440266 },
		{ 116.81329322863975, -116.1371994627773 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -0.9822783063440266 },
		{ 116.81329322863975, -116.1371994627773 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -0.9822783063440266 },
		{ 116.81329322863975, -116.1371994627773 },
		{ 0.0 },
		{ 0.0 }
	}
};

TF1_INF	gstPIInf[6] = {
	{
		{ 1.0, -1.0 },
		{ 2.8921552789050216, -2.8487290195402286 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -1.0 },
		{ 2.8921552789050216, -2.8487290195402286 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -1.0 },
		{ 2.8921552789050216, -2.8487290195402286 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -1.0 },
		{ 2.8921552789050216, -2.8487290195402286 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -1.0 },
		{ 2.8921552789050216, -2.8487290195402286 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -1.0 },
		{ 2.8921552789050216, -2.8487290195402286 },
		{ 0.0 },
		{ 0.0 }
	}
};

TF2_INF	gstNFInf[6][1] = {
	{
		{
			{ 1.0, -0.09024972912144005, 0.3896611373753469 },
			{ 0.6527719615970552, -9.184981609578122e-06, 0.6466486316386724 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	},
	{
		{
			{ 1.0, -0.09024972912144005, 0.3896611373753469 },
			{ 0.6527719615970552, -9.184981609578122e-06, 0.6466486316386724 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	},
	{
		{
			{ 1.0, -0.09024972912144005, 0.3896611373753469 },
			{ 0.6527719615970552, -9.184981609578122e-06, 0.6466486316386724 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	},
	{
		{
			{ 1.0, -0.09024972912144005, 0.3896611373753469 },
			{ 0.6527719615970552, -9.184981609578122e-06, 0.6466486316386724 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	},
	{
		{
			{ 1.0, -0.09024972912144005, 0.3896611373753469 },
			{ 0.6527719615970552, -9.184981609578122e-06, 0.6466486316386724 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	},
	{
		{
			{ 1.0, -0.09024972912144005, 0.3896611373753469 },
			{ 0.6527719615970552, -9.184981609578122e-06, 0.6466486316386724 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	}
};

TF2_INF	gstPFInf[6][9] = {
	{
		{
			{ 1.0, -1.9999226088105237, 0.9999842924023302 },
			{ 0.004846029932398133, -0.009070802086785124, 0.00422477215438688 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9997218654890982, 0.9999685865046481 },
			{ 0.006276369710432772, -0.012413867743413354, 0.006137498032980693 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9993978086303652, 0.9999528837596715 },
			{ 0.06376199777331581, -0.12706089396131004, 0.06329889618799411 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.999358622866345, 0.9999513137144745 },
			{ 0.012204074864599512, -0.02433430036229356, 0.012130225497694047 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9983800226645165, 0.9999214935333114 },
			{ 0.019937463902848895, -0.04029876518600961, 0.020361301283160715 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9976864846353204, 0.9999058089506381 },
			{ 0.022735553049226365, -0.04643277152027081, 0.023697218471044446 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.996870015135926, 0.9998901333171952 },
			{ 0.02454388843701938, -0.05082722790884997, 0.026283339471830813 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9948689131501718, 0.9998588146680686 },
			{ 0.024702365636649626, -0.05356076604629534, 0.0288584004096456 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9798349131200155, 0.9997187019309638 },
			{ -0.056256406187663144, 0.07794288357681367, -0.021686477389150305 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	},
	{
		{
			{ 1.0, -1.9999226088105237, 0.9999842924023302 },
			{ 0.004846029932398133, -0.009070802086785124, 0.00422477215438688 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9997218654890982, 0.9999685865046481 },
			{ 0.006276369710432772, -0.012413867743413354, 0.006137498032980693 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9993978086303652, 0.9999528837596715 },
			{ 0.06376199777331581, -0.12706089396131004, 0.06329889618799411 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.999358622866345, 0.9999513137144745 },
			{ 0.012204074864599512, -0.02433430036229356, 0.012130225497694047 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9983800226645165, 0.9999214935333114 },
			{ 0.019937463902848895, -0.04029876518600961, 0.020361301283160715 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9976864846353204, 0.9999058089506381 },
			{ 0.022735553049226365, -0.04643277152027081, 0.023697218471044446 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.996870015135926, 0.9998901333171952 },
			{ 0.02454388843701938, -0.05082722790884997, 0.026283339471830813 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9948689131501718, 0.9998588146680686 },
			{ 0.024702365636649626, -0.05356076604629534, 0.0288584004096456 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9798349131200155, 0.9997187019309638 },
			{ -0.056256406187663144, 0.07794288357681367, -0.021686477389150305 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	},
	{
		{
			{ 1.0, -1.9999226088105237, 0.9999842924023302 },
			{ 0.004846029932398133, -0.009070802086785124, 0.00422477215438688 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9997218654890982, 0.9999685865046481 },
			{ 0.006276369710432772, -0.012413867743413354, 0.006137498032980693 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9993978086303652, 0.9999528837596715 },
			{ 0.06376199777331581, -0.12706089396131004, 0.06329889618799411 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.999358622866345, 0.9999513137144745 },
			{ 0.012204074864599512, -0.02433430036229356, 0.012130225497694047 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9983800226645165, 0.9999214935333114 },
			{ 0.019937463902848895, -0.04029876518600961, 0.020361301283160715 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9976864846353204, 0.9999058089506381 },
			{ 0.022735553049226365, -0.04643277152027081, 0.023697218471044446 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.996870015135926, 0.9998901333171952 },
			{ 0.02454388843701938, -0.05082722790884997, 0.026283339471830813 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9948689131501718, 0.9998588146680686 },
			{ 0.024702365636649626, -0.05356076604629534, 0.0288584004096456 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9798349131200155, 0.9997187019309638 },
			{ -0.056256406187663144, 0.07794288357681367, -0.021686477389150305 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	},
	{
		{
			{ 1.0, -1.9999226088105237, 0.9999842924023302 },
			{ 0.004846029932398133, -0.009070802086785124, 0.00422477215438688 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9997218654890982, 0.9999685865046481 },
			{ 0.006276369710432772, -0.012413867743413354, 0.006137498032980693 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9993978086303652, 0.9999528837596715 },
			{ 0.06376199777331581, -0.12706089396131004, 0.06329889618799411 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.999358622866345, 0.9999513137144745 },
			{ 0.012204074864599512, -0.02433430036229356, 0.012130225497694047 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9983800226645165, 0.9999214935333114 },
			{ 0.019937463902848895, -0.04029876518600961, 0.020361301283160715 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9976864846353204, 0.9999058089506381 },
			{ 0.022735553049226365, -0.04643277152027081, 0.023697218471044446 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.996870015135926, 0.9998901333171952 },
			{ 0.02454388843701938, -0.05082722790884997, 0.026283339471830813 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9948689131501718, 0.9998588146680686 },
			{ 0.024702365636649626, -0.05356076604629534, 0.0288584004096456 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9798349131200155, 0.9997187019309638 },
			{ -0.056256406187663144, 0.07794288357681367, -0.021686477389150305 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	},
	{
		{
			{ 1.0, -1.9999226088105237, 0.9999842924023302 },
			{ 0.004846029932398133, -0.009070802086785124, 0.00422477215438688 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9997218654890982, 0.9999685865046481 },
			{ 0.006276369710432772, -0.012413867743413354, 0.006137498032980693 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9993978086303652, 0.9999528837596715 },
			{ 0.06376199777331581, -0.12706089396131004, 0.06329889618799411 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.999358622866345, 0.9999513137144745 },
			{ 0.012204074864599512, -0.02433430036229356, 0.012130225497694047 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9983800226645165, 0.9999214935333114 },
			{ 0.019937463902848895, -0.04029876518600961, 0.020361301283160715 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9976864846353204, 0.9999058089506381 },
			{ 0.022735553049226365, -0.04643277152027081, 0.023697218471044446 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.996870015135926, 0.9998901333171952 },
			{ 0.02454388843701938, -0.05082722790884997, 0.026283339471830813 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9948689131501718, 0.9998588146680686 },
			{ 0.024702365636649626, -0.05356076604629534, 0.0288584004096456 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9798349131200155, 0.9997187019309638 },
			{ -0.056256406187663144, 0.07794288357681367, -0.021686477389150305 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	},
	{
		{
			{ 1.0, -1.9999226088105237, 0.9999842924023302 },
			{ 0.004846029932398133, -0.009070802086785124, 0.00422477215438688 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9997218654890982, 0.9999685865046481 },
			{ 0.006276369710432772, -0.012413867743413354, 0.006137498032980693 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9993978086303652, 0.9999528837596715 },
			{ 0.06376199777331581, -0.12706089396131004, 0.06329889618799411 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.999358622866345, 0.9999513137144745 },
			{ 0.012204074864599512, -0.02433430036229356, 0.012130225497694047 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9983800226645165, 0.9999214935333114 },
			{ 0.019937463902848895, -0.04029876518600961, 0.020361301283160715 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9976864846353204, 0.9999058089506381 },
			{ 0.022735553049226365, -0.04643277152027081, 0.023697218471044446 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.996870015135926, 0.9998901333171952 },
			{ 0.02454388843701938, -0.05082722790884997, 0.026283339471830813 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9948689131501718, 0.9998588146680686 },
			{ 0.024702365636649626, -0.05356076604629534, 0.0288584004096456 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		},
		{
			{ 1.0, -1.9798349131200155, 0.9997187019309638 },
			{ -0.056256406187663144, 0.07794288357681367, -0.021686477389150305 },
			{ 0.0, 0.0 },
			{ 0.0, 0.0 }
		}
	}
};

TF2_INF	gstDOBfbuInf[6] = {
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 0.0, 0.0029206149947564874, 0.0029178505701443758 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 0.0, 0.0029206149947564874, 0.0029178505701443758 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 0.0, 0.0029206149947564874, 0.0029178505701443758 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 0.0, 0.0029206149947564874, 0.0029178505701443758 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 0.0, 0.0029206149947564874, 0.0029178505701443758 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 0.0, 0.0029206149947564874, 0.0029178505701443758 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	}
};

TF2_INF	gstDOBfbyInf[6] = {
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 8232.241975359282, -16441.13008845896, 8208.888113099661 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 8232.241975359282, -16441.13008845896, 8208.888113099661 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 8232.241975359282, -16441.13008845896, 8208.888113099661 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 8232.241975359282, -16441.13008845896, 8208.888113099661 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 8232.241975359282, -16441.13008845896, 8208.888113099661 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.8900353176517015, 0.8958737832166023 },
		{ 8232.241975359282, -16441.13008845896, 8208.888113099661 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	}
};

TF2_INF	gstDOBestuInf[6] = {
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 0.0, 0.022053798396028945, 0.022012069377029597 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 0.0, 0.022053798396028945, 0.022012069377029597 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 0.0, 0.022053798396028945, 0.022012069377029597 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 0.0, 0.022053798396028945, 0.022012069377029597 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 0.0, 0.022053798396028945, 0.022012069377029597 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 0.0, 0.022053798396028945, 0.022012069377029597 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	}
};

TF2_INF	gstDOBestyInf[6] = {
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 62221.172211634796, -124089.81748108512, 61868.64526940961 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 62221.172211634796, -124089.81748108512, 61868.64526940961 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 62221.172211634796, -124089.81748108512, 61868.64526940961 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 62221.172211634796, -124089.81748108512, 61868.64526940961 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 62221.172211634796, -124089.81748108512, 61868.64526940961 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.580162565875511, 0.6242284336485696 },
		{ 62221.172211634796, -124089.81748108512, 61868.64526940961 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	}
};

TF3_INF	gstZPETInf[6] = {
	{
		{ 1.0, 0.0, 0.0, 0.0 },
		{ 706670.4560019532, -704003.7920687284, -704003.7838673632, 701337.1199341384 },
		{ 0.0, 0.0, 0.0 },
		{ 0.0, 0.0, 0.0 }
	},
	{
		{ 1.0, 0.0, 0.0, 0.0 },
		{ 706670.4560019532, -704003.7920687284, -704003.7838673632, 701337.1199341384 },
		{ 0.0, 0.0, 0.0 },
		{ 0.0, 0.0, 0.0 }
	},
	{
		{ 1.0, 0.0, 0.0, 0.0 },
		{ 706670.4560019532, -704003.7920687284, -704003.7838673632, 701337.1199341384 },
		{ 0.0, 0.0, 0.0 },
		{ 0.0, 0.0, 0.0 }
	},
	{
		{ 1.0, 0.0, 0.0, 0.0 },
		{ 706670.4560019532, -704003.7920687284, -704003.7838673632, 701337.1199341384 },
		{ 0.0, 0.0, 0.0 },
		{ 0.0, 0.0, 0.0 }
	},
	{
		{ 1.0, 0.0, 0.0, 0.0 },
		{ 706670.4560019532, -704003.7920687284, -704003.7838673632, 701337.1199341384 },
		{ 0.0, 0.0, 0.0 },
		{ 0.0, 0.0, 0.0 }
	},
	{
		{ 1.0, 0.0, 0.0, 0.0 },
		{ 706670.4560019532, -704003.7920687284, -704003.7838673632, 701337.1199341384 },
		{ 0.0, 0.0, 0.0 },
		{ 0.0, 0.0, 0.0 }
	}
};

TF2_INF	gstImpInf[6] = {
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	},
	{
		{ 1.0, -1.9943342928194054, 0.9943342928194054 },
		{ 0.0, 3.54442026839763e-07, 3.537713706425194e-07 },
		{ 0.0, 0.0 },
		{ 0.0, 0.0 }
	}
};

TF1_INF	gstHapInf[6] = {
	{
		{ 1.0, -0.5783006173742605 },
		{ 35906.010371427954, -34743.26189053057 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -0.5783006173742605 },
		{ 35906.010371427954, -34743.26189053057 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -0.5783006173742605 },
		{ 35906.010371427954, -34743.26189053057 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -0.5783006173742605 },
		{ 35906.010371427954, -34743.26189053057 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -0.5783006173742605 },
		{ 35906.010371427954, -34743.26189053057 },
		{ 0.0 },
		{ 0.0 }
	},
	{
		{ 1.0, -0.5783006173742605 },
		{ 35906.010371427954, -34743.26189053057 },
		{ 0.0 },
		{ 0.0 }
	}
};